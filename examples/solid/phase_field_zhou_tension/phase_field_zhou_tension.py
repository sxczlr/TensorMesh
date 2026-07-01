"""
Zhou-Rabczuk-Zhuang single-edge-notched tension benchmark
=========================================================

TensorMesh validation model for the 2D single-edge-notched square plate under
quasi-static tension from:

Zhou S., Rabczuk T., Zhuang X. Phase field modeling of quasi-static and dynamic
crack propagation: COMSOL implementation and case studies. Advances in
Engineering Software, 2018, 122: 31-49.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
TENSORMESH_ROOT = next(parent for parent in SCRIPT_DIR.parents if (parent / "tensormesh").is_dir())
sys.path.insert(0, str(TENSORMESH_ROOT))

from tensormesh import Mesh
from tensormesh.assemble import ElementAssembler
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.element import element_type2element, element_type2order
from tensormesh.sparse import spsolve


# Paper units are mm, N, and N/mm^2.
GEOMETRY = {
    "left": 0.0,
    "right": 1.0,
    "bottom": 0.0,
    "top": 1.0,
    "notch_start": (0.0, 0.5),
    "notch_end": (0.5, 0.5),
}
MATERIAL = {
    "E": 210.0e3,  # 210 GPa = 210000 N/mm^2
    "nu": 0.3,
    "Gc": 2.7,  # 2700 J/m^2 = 2.7 N/mm
    "k": 1.0e-9,
}
PAPER_NUMERICS = {
    "l0_small": 7.5e-3,
    "l0_large": 1.5e-2,
    "max_h_over_l0": 0.5,
    "paper_h": 3.96e-3,
    "B": 1.0e6,
    "tolerance": 1.0e-6,
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def paper_displacement_schedule(max_steps: int | None, final_displacement: float) -> list[float]:
    values: list[float] = []
    value = 0.0
    while value < min(450.0e-5, final_displacement) - 1.0e-15:
        value += 1.0e-5
        values.append(min(value, final_displacement))
        if max_steps is not None and len(values) >= max_steps:
            return values

    while value < final_displacement - 1.0e-15:
        value += 1.0e-6
        values.append(min(value, final_displacement))
        if max_steps is not None and len(values) >= max_steps:
            return values

    return values


def uniform_displacement_schedule(n_steps: int, final_displacement: float) -> list[float]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    step = final_displacement / n_steps
    return [step * i for i in range(1, n_steps + 1)]


def build_mesh(args: argparse.Namespace) -> tuple[Mesh, str]:
    if args.mesh_file:
        mesh_path = Path(args.mesh_file)
        mesh = Mesh.from_file(str(mesh_path), reorder=False)
        return mesh, str(mesh_path)

    cache_dir = SCRIPT_DIR / "_mesh_cache"
    cache_path = cache_dir / f"zhou_tension_h{args.chara_length:g}_{args.element_type}.msh"
    mesh = gen_rectangle(
        left=GEOMETRY["left"],
        right=GEOMETRY["right"],
        bottom=GEOMETRY["bottom"],
        top=GEOMETRY["top"],
        chara_length=args.chara_length,
        order=1,
        element_type=args.element_type,
        cache_path=str(cache_path),
    )
    return mesh, str(cache_path)


def max_allowed_chara_length(length_scale: float) -> float:
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive.")
    return PAPER_NUMERICS["max_h_over_l0"] * length_scale


def enforce_phase_field_mesh_resolution(args: argparse.Namespace) -> None:
    """Keep the generated mesh fine enough to resolve the phase-field band."""
    args.requested_chara_length = args.chara_length
    args.max_chara_length = max_allowed_chara_length(args.length_scale)
    if getattr(args, "mesh_file", None) is None and args.chara_length > args.max_chara_length:
        args.chara_length = args.max_chara_length


def strain_spectral_energy(
    grad_u: torch.Tensor,
    *,
    E: float,
    nu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensile and compressive plane-strain elastic energy densities."""
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    strain_2d = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    eps_xx = strain_2d[..., 0, 0]
    eps_yy = strain_2d[..., 1, 1]
    eps_xy = strain_2d[..., 0, 1]

    mean = 0.5 * (eps_xx + eps_yy)
    spectral_eps = 1.0e-12 if strain_2d.dtype == torch.float32 else 1.0e-24
    radius = torch.sqrt(torch.clamp((0.5 * (eps_xx - eps_yy)) ** 2 + eps_xy**2, min=0.0) + spectral_eps)
    principal = torch.stack((mean - radius, mean + radius, torch.zeros_like(mean)), dim=-1)

    principal_pos = torch.relu(principal)
    principal_neg = -torch.relu(-principal)
    trace = eps_xx + eps_yy
    trace_pos = torch.relu(trace)
    trace_neg = -torch.relu(-trace)

    psi_pos = 0.5 * lmbda * trace_pos**2 + mu * (principal_pos * principal_pos).sum(dim=-1)
    psi_neg = 0.5 * lmbda * trace_neg**2 + mu * (principal_neg * principal_neg).sum(dim=-1)
    return psi_pos, psi_neg


class ZhouTensionPhaseFieldModel(ElementAssembler):
    """Energy forms used by the Zhou et al. quasi-static tension benchmark."""

    def __post_init__(self, E: float, nu: float, Gc: float, l0: float, k: float) -> None:
        self.E = float(E)
        self.nu = float(nu)
        self.Gc = float(Gc)
        self.l0 = float(l0)
        self.k = float(k)

    def forward(self, u, v):
        return u * v

    def tensile_energy_density(self, graddisplacement: torch.Tensor) -> torch.Tensor:
        psi_pos, _ = strain_spectral_energy(graddisplacement, E=self.E, nu=self.nu)
        return psi_pos

    def elastic_density(self, graddisplacement: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        psi_pos, psi_neg = strain_spectral_energy(graddisplacement, E=self.E, nu=self.nu)
        degradation = (1.0 - self.k) * (1.0 - phase.squeeze()) ** 2 + self.k
        return degradation * psi_pos + psi_neg

    def fracture_density(self, phase: torch.Tensor, gradphase: torch.Tensor) -> torch.Tensor:
        phi = phase.squeeze()
        return self.Gc * (phi**2 / (2.0 * self.l0) + 0.5 * self.l0 * (gradphase * gradphase).sum())

    def phase_history_density(self, phase: torch.Tensor, gradphase: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        phi = phase.squeeze()
        return self.fracture_density(phase, gradphase) + (1.0 - self.k) * history * (1.0 - phi) ** 2

    def element_energy(self, graddisplacement: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.elastic_density(graddisplacement, phase)


def boundary_masks(points: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = points.dtype
    tol = 1.0e-9 if dtype == torch.float64 else 1.0e-6
    return {
        "left": torch.isclose(points[:, 0], torch.as_tensor(GEOMETRY["left"], dtype=dtype, device=points.device), atol=tol),
        "right": torch.isclose(points[:, 0], torch.as_tensor(GEOMETRY["right"], dtype=dtype, device=points.device), atol=tol),
        "bottom": torch.isclose(points[:, 1], torch.as_tensor(GEOMETRY["bottom"], dtype=dtype, device=points.device), atol=tol),
        "top": torch.isclose(points[:, 1], torch.as_tensor(GEOMETRY["top"], dtype=dtype, device=points.device), atol=tol),
    }


def apply_tension_bc(
    displacement: torch.Tensor,
    masks: dict[str, torch.Tensor],
    prescribed_u: float,
    fix_side_x: bool,
) -> torch.Tensor:
    mask = torch.ones_like(displacement)
    values = torch.zeros_like(displacement)

    mask[masks["bottom"], :] = 0.0
    mask[masks["top"], 0] = 0.0
    mask[masks["top"], 1] = 0.0
    values[masks["top"], 1] = prescribed_u

    if fix_side_x:
        side = masks["left"] | masks["right"]
        mask[side, 0] = 0.0

    return displacement * mask + values


def segment_distance(points: torch.Tensor) -> torch.Tensor:
    start = torch.as_tensor(GEOMETRY["notch_start"], dtype=points.dtype, device=points.device)
    end = torch.as_tensor(GEOMETRY["notch_end"], dtype=points.dtype, device=points.device)
    tangent = end - start
    length_sq = (tangent * tangent).sum()
    t = ((points - start) * tangent).sum(dim=-1) / length_sq
    t = torch.clamp(t, 0.0, 1.0)
    closest = start + t[..., None] * tangent
    return torch.linalg.norm(points - closest, dim=-1)


def initial_history_from_points(points: torch.Tensor, Gc: float, l0: float, B: float) -> torch.Tensor:
    distance = segment_distance(points)
    history = B * Gc / (2.0 * l0) * (1.0 - 2.0 * distance / l0)
    return torch.where(distance <= 0.5 * l0, history, torch.zeros_like(history))


def initial_phase_from_points(points: torch.Tensor, Gc: float, l0: float, B: float, k: float) -> torch.Tensor:
    history = initial_history_from_points(points, Gc, l0, B)
    numerator = 2.0 * l0 * (1.0 - k) * history / Gc
    return torch.clamp(numerator / (1.0 + numerator), 0.0, 1.0)


def quadrature_points(model: ZhouTensionPhaseFieldModel) -> dict[str, torch.Tensor]:
    q_points = {}
    points = next(iter(model.transformation.values())).points
    for element_type in model.element_types:
        elements = model.elements[element_type]
        trans = model.transformation[element_type]
        q_points[element_type] = torch.einsum("ebd,qb->eqd", points[elements], trans.shape_val)
    return q_points


def initial_history_at_quadrature(model: ZhouTensionPhaseFieldModel, B: float) -> dict[str, torch.Tensor]:
    return {
        element_type: initial_history_from_points(q, model.Gc, model.l0, B)
        for element_type, q in quadrature_points(model).items()
    }


def tensile_energy_at_quadrature(
    model: ZhouTensionPhaseFieldModel,
    displacement: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for element_type in model.element_types:
        elements = model.elements[element_type]
        trans = model.transformation[element_type]
        u_elem = displacement[elements]
        grad_u = torch.einsum("ebc,eqbd->eqcd", u_elem, trans.shape_grad)
        psi_pos, _ = strain_spectral_energy(grad_u, E=model.E, nu=model.nu)
        values[element_type] = psi_pos.detach()
    return values


def max_history(
    old_history: dict[str, torch.Tensor],
    new_history: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: torch.maximum(old_history[key], new_history[key]) for key in old_history}


def coalesce_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = row * shape[1] + col
    unique_flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    unique_values = torch.zeros(unique_flat.shape[0], dtype=values.dtype, device=values.device)
    unique_values.index_add_(0, inverse, values)
    unique_row = torch.div(unique_flat, shape[1], rounding_mode="floor")
    unique_col = unique_flat - unique_row * shape[1]
    return unique_row.long(), unique_col.long(), unique_values


def structured_grid_laplacian(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    x_values = torch.sort(torch.unique(points[:, 0].detach())).values
    y_values = torch.sort(torch.unique(points[:, 1].detach())).values
    nx = int(x_values.numel())
    ny = int(y_values.numel())
    n_points = int(points.shape[0])
    if nx * ny != n_points or nx < 2 or ny < 2:
        return None

    lookup = {
        (round(float(points[i, 0].detach().cpu()), 12), round(float(points[i, 1].detach().cpu()), 12)): i
        for i in range(n_points)
    }
    grid = torch.empty((ny, nx), dtype=torch.long, device=points.device)
    for iy, y in enumerate(y_values.detach().cpu().tolist()):
        for ix, x in enumerate(x_values.detach().cpu().tolist()):
            index = lookup.get((round(float(x), 12), round(float(y), 12)))
            if index is None:
                return None
            grid[iy, ix] = index

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(row_idx: int, col_idx: int, value: float) -> None:
        rows.append(row_idx)
        cols.append(col_idx)
        vals.append(value)

    x_cpu = x_values.detach().cpu()
    y_cpu = y_values.detach().cpu()
    grid_cpu = grid.detach().cpu()
    for iy in range(ny):
        for ix in range(nx):
            center = int(grid_cpu[iy, ix])
            diag = 0.0

            if ix == 0:
                coeff = 2.0 / float((x_cpu[1] - x_cpu[0]) ** 2)
                add(center, int(grid_cpu[iy, ix + 1]), coeff)
                diag -= coeff
            elif ix == nx - 1:
                coeff = 2.0 / float((x_cpu[-1] - x_cpu[-2]) ** 2)
                add(center, int(grid_cpu[iy, ix - 1]), coeff)
                diag -= coeff
            else:
                h_left = float(x_cpu[ix] - x_cpu[ix - 1])
                h_right = float(x_cpu[ix + 1] - x_cpu[ix])
                left_coeff = 2.0 / (h_left * (h_left + h_right))
                right_coeff = 2.0 / (h_right * (h_left + h_right))
                add(center, int(grid_cpu[iy, ix - 1]), left_coeff)
                add(center, int(grid_cpu[iy, ix + 1]), right_coeff)
                diag -= left_coeff + right_coeff

            if iy == 0:
                coeff = 2.0 / float((y_cpu[1] - y_cpu[0]) ** 2)
                add(center, int(grid_cpu[iy + 1, ix]), coeff)
                diag -= coeff
            elif iy == ny - 1:
                coeff = 2.0 / float((y_cpu[-1] - y_cpu[-2]) ** 2)
                add(center, int(grid_cpu[iy - 1, ix]), coeff)
                diag -= coeff
            else:
                h_down = float(y_cpu[iy] - y_cpu[iy - 1])
                h_up = float(y_cpu[iy + 1] - y_cpu[iy])
                down_coeff = 2.0 / (h_down * (h_down + h_up))
                up_coeff = 2.0 / (h_up * (h_down + h_up))
                add(center, int(grid_cpu[iy - 1, ix]), down_coeff)
                add(center, int(grid_cpu[iy + 1, ix]), up_coeff)
                diag -= down_coeff + up_coeff

            add(center, center, diag)

    row = torch.as_tensor(rows, dtype=torch.long, device=points.device)
    col = torch.as_tensor(cols, dtype=torch.long, device=points.device)
    values = torch.as_tensor(vals, dtype=points.dtype, device=points.device)
    return coalesce_coo(row, col, values, (n_points, n_points))


def mesh_edge_pairs(mesh: Mesh) -> torch.Tensor:
    all_pairs = []
    elements_by_type = mesh.elements(-1)
    for element_type, elements in elements_by_type.items():
        element = element_type2element(element_type)
        order = element_type2order[element_type]
        facets = element.get_facet(order)
        if isinstance(facets, tuple):
            facets = torch.cat([facet for facet in facets if facet.shape[-1] >= 2], dim=0)
        local_pairs = []
        for facet in facets:
            local_pairs.append(torch.stack((facet[:-1], facet[1:]), dim=-1))
        if not local_pairs:
            continue
        local_pairs_tensor = torch.cat(local_pairs, dim=0).to(device=elements.device)
        pairs = elements[:, local_pairs_tensor].reshape(-1, 2)
        all_pairs.append(pairs)

    if not all_pairs:
        raise RuntimeError("Could not build phase-field graph Laplacian because the mesh has no edges.")
    pairs = torch.cat(all_pairs, dim=0)
    pairs = torch.cat((pairs, pairs.flip(dims=(1,))), dim=0)
    return torch.unique(pairs, sorted=True, dim=0)


def graph_laplacian(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = mesh_edge_pairs(mesh)
    row = edges[:, 0]
    col = edges[:, 1]
    distance_sq = ((mesh.points[row] - mesh.points[col]) ** 2).sum(dim=-1)
    keep = distance_sq > torch.finfo(mesh.points.dtype).eps
    row = row[keep]
    col = col[keep]
    weights = 1.0 / distance_sq[keep]

    diag = torch.zeros(mesh.n_points, dtype=mesh.points.dtype, device=mesh.points.device)
    diag.index_add_(0, row, -weights)
    diag_index = torch.arange(mesh.n_points, dtype=torch.long, device=mesh.points.device)
    lap_row = torch.cat((row, diag_index))
    lap_col = torch.cat((col, diag_index))
    lap_values = torch.cat((weights, diag))
    return coalesce_coo(lap_row, lap_col, lap_values, (mesh.n_points, mesh.n_points))


def build_phase_laplacian(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    structured = structured_grid_laplacian(mesh.points)
    if structured is not None:
        row, col, values = structured
        return row, col, values, "structured_finite_difference_neumann"
    row, col, values = graph_laplacian(mesh)
    return row, col, values, "edge_graph_laplacian"


def history_at_nodes(model: ZhouTensionPhaseFieldModel, history: dict[str, torch.Tensor]) -> torch.Tensor:
    points = next(iter(model.transformation.values())).points
    numerator = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    denominator = torch.zeros_like(numerator)

    for element_type in model.element_types:
        elements = model.elements[element_type]
        element_history = history[element_type].mean(dim=1)
        expanded_history = element_history[:, None].expand_as(elements).reshape(-1)
        numerator.index_add_(0, elements.reshape(-1), expanded_history)
        denominator.index_add_(0, elements.reshape(-1), torch.ones_like(expanded_history))

    return numerator / torch.clamp(denominator, min=1.0)


def model_energy(
    model: ZhouTensionPhaseFieldModel,
    displacement: torch.Tensor,
    phase: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    return model.energy(point_data={"displacement": displacement, "phase": phase}, batch_size=batch_size)


def fracture_energy(model: ZhouTensionPhaseFieldModel, phase: torch.Tensor, batch_size: int) -> torch.Tensor:
    return model.energy(point_data={"phase": phase}, func=model.fracture_density, batch_size=batch_size)


def solve_solid_mechanics_step(
    model: ZhouTensionPhaseFieldModel,
    displacement: torch.Tensor,
    phase: torch.Tensor,
    masks: dict[str, torch.Tensor],
    prescribed_u: float,
    fix_side_x: bool,
    u_iters: int,
    batch_size: int,
) -> None:
    """Solve the displacement field with the phase field fixed."""
    phase_fixed = phase.detach()
    optimizer = optim.LBFGS(
        [displacement],
        lr=1.0,
        max_iter=u_iters,
        max_eval=max(u_iters + 4, u_iters),
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        u_active = apply_tension_bc(displacement, masks, prescribed_u, fix_side_x)
        loss = model_energy(model, u_active, phase_fixed, batch_size)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        u_active = apply_tension_bc(displacement, masks, prescribed_u, fix_side_x)
        displacement.copy_(u_active)


def solve_phase_field_step(
    model: ZhouTensionPhaseFieldModel,
    phase: torch.Tensor,
    history: dict[str, torch.Tensor],
    previous_phase: torch.Tensor,
    laplace_row: torch.Tensor,
    laplace_col: torch.Tensor,
    laplace_values: torch.Tensor,
) -> None:
    """Solve the strong-form AT2 phase-field equation with fixed history."""
    h_node = history_at_nodes(model, history).detach()
    n_points = phase.shape[0]
    diag_index = torch.arange(n_points, dtype=torch.long, device=phase.device)
    reaction_diag = model.Gc / model.l0 + 2.0 * (1.0 - model.k) * h_node
    rhs = 2.0 * (1.0 - model.k) * h_node

    row = torch.cat((laplace_row, diag_index))
    col = torch.cat((laplace_col, diag_index))
    values = torch.cat((-model.Gc * model.l0 * laplace_values, reaction_diag))
    row, col, values = coalesce_coo(row, col, values, (n_points, n_points))
    phi_new = spsolve(
        values,
        row,
        col,
        (n_points, n_points),
        rhs,
        method="cg",
        preconditioner="jacobi",
        tol=1.0e-10,
        max_iter=20000,
        is_spd=True,
    )

    with torch.no_grad():
        phase.copy_(phi_new.reshape_as(phase))
        phase.clamp_(0.0, 1.0)
        phase.copy_(torch.maximum(phase, previous_phase))


def compute_reaction(
    model: ZhouTensionPhaseFieldModel,
    displacement: torch.Tensor,
    phase: torch.Tensor,
    top_mask: torch.Tensor,
    batch_size: int,
) -> float:
    trial_u = displacement.detach().clone().requires_grad_(True)
    total_energy = model_energy(model, trial_u, phase.detach(), batch_size)
    grad_u = torch.autograd.grad(total_energy, trial_u)[0]
    return float(grad_u[top_mask, 1].sum().detach().cpu())


def crack_metrics(
    mesh: Mesh,
    phase: torch.Tensor,
    threshold: float,
    band: float,
    through_tol: float,
) -> dict[str, float | bool]:
    points = mesh.points.detach()
    phi = phase.detach()
    notch_y = torch.as_tensor(GEOMETRY["notch_start"][1], dtype=points.dtype, device=points.device)
    right = float(GEOMETRY["right"])
    band_mask = torch.abs(points[:, 1] - notch_y) <= band
    crack_mask = (phi >= threshold) & band_mask

    if torch.any(crack_mask):
        tip_x = float(points[crack_mask, 0].max().cpu())
    else:
        tip_x = float(GEOMETRY["notch_end"][0])

    length = max(0.0, tip_x - float(GEOMETRY["notch_start"][0]))
    return {
        "crack_tip_x_mm": tip_x,
        "crack_length_mm": length,
        "is_developed": tip_x > float(GEOMETRY["notch_end"][0]) + band * 0.25,
        "is_through": tip_x >= right - through_tol,
    }


def relative_change(new: torch.Tensor, old: torch.Tensor) -> float:
    num = torch.linalg.norm((new - old).reshape(-1))
    den = torch.linalg.norm(old.reshape(-1)) + 1.0e-14
    return float((num / den).detach().cpu())


def save_phase_plot(
    mesh: Mesh,
    phase: torch.Tensor,
    displacement: torch.Tensor,
    output_file: Path,
    title_suffix: str = "",
) -> None:
    points = mesh.points.detach().cpu().numpy()
    phase_np = phase.detach().cpu().numpy()
    disp_np = displacement.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    im0 = axes[0].tripcolor(tri, phase_np, shading="gouraud", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].plot([0.0, 0.5], [0.5, 0.5], "c-", linewidth=1.3)
    axes[0].set_title("phase field phi" if not title_suffix else f"phase field phi\n{title_suffix}")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    deformed = points + disp_np
    tri_def = mtri.Triangulation(deformed[:, 0], deformed[:, 1])
    disp_norm = torch.linalg.norm(torch.as_tensor(disp_np), dim=1).numpy()
    im1 = axes[1].tripcolor(tri_def, disp_norm, shading="gouraud", cmap="viridis")
    axes[1].set_title("deformed displacement norm" if not title_suffix else f"deformed displacement norm\n{title_suffix}")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=240)
    plt.close(fig)


def save_loading_overview(mesh: Mesh, snapshots: list[dict], output_file: Path) -> None:
    if not snapshots:
        return

    points = mesh.points.detach().cpu().numpy()
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    n_steps = len(snapshots)
    width = max(3.0 * n_steps, 6.0)
    fig, axes = plt.subplots(2, n_steps, figsize=(width, 6.0), squeeze=False, constrained_layout=True)

    max_disp = max(float(s["disp_norm"].max()) for s in snapshots)
    for col, snap in enumerate(snapshots):
        phase = snap["phase"]
        disp_norm = snap["disp_norm"]
        reaction_label = f"{snap['reaction']:.2e}" if math.isfinite(snap["reaction"]) else "not sampled"
        label = f"step {snap['step']}\nu={snap['prescribed']:.1e} mm\nR={reaction_label}"

        im_phase = axes[0, col].tripcolor(tri, phase, shading="gouraud", cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, col].plot([0.0, 0.5], [0.5, 0.5], "c-", linewidth=1.0)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].set_aspect("equal")
        axes[0, col].axis("off")

        im_disp = axes[1, col].tripcolor(tri, disp_norm, shading="gouraud", cmap="viridis", vmin=0.0, vmax=max_disp)
        axes[1, col].set_aspect("equal")
        axes[1, col].axis("off")

    fig.colorbar(im_phase, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.01, label="phi")
    fig.colorbar(im_disp, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.01, label="|u| [mm]")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=240)
    plt.close(fig)


def write_csv(rows: list[dict[str, float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_load_curve(rows: list[dict[str, float]], output_file: Path) -> None:
    sampled = [row for row in rows if math.isfinite(row["reaction_y_N_per_thickness"])]
    x = [row["prescribed_displacement_mm"] for row in sampled]
    y = [row["reaction_y_N_per_thickness"] for row in sampled]
    x_tip = [row["prescribed_displacement_mm"] for row in rows]
    tip = [row["crack_tip_x_mm"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    if x:
        axes[0].plot(x, y, marker="o", markersize=2.5, linewidth=1.4)
    axes[0].set_xlabel("prescribed displacement [mm]")
    axes[0].set_ylabel("reaction force [N per thickness]")
    axes[0].set_title("load-displacement response")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_tip, tip, marker="o", markersize=2.5, linewidth=1.4, color="tab:red")
    axes[1].axhline(GEOMETRY["right"], color="0.3", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("prescribed displacement [mm]")
    axes[1].set_ylabel("crack tip x [mm]")
    axes[1].set_title("crack advance")
    axes[1].grid(True, alpha=0.3)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=240)
    plt.close(fig)


def validate_summary(summary: dict) -> list[str]:
    checks = []
    mesh = summary["mesh"]
    if abs(mesh["x_min"] - GEOMETRY["left"]) < 1.0e-8 and abs(mesh["x_max"] - GEOMETRY["right"]) < 1.0e-8:
        checks.append("domain_x_ok")
    if abs(mesh["y_min"] - GEOMETRY["bottom"]) < 1.0e-8 and abs(mesh["y_max"] - GEOMETRY["top"]) < 1.0e-8:
        checks.append("domain_y_ok")
    if summary["max_top_uy_error"] < 1.0e-8:
        checks.append("top_uy_bc_ok")
    if summary["max_top_ux_error"] < 1.0e-8:
        checks.append("top_ux_bc_ok")
    if summary["max_bottom_displacement_error"] < 1.0e-8:
        checks.append("bottom_bc_ok")
    if 0.0 <= summary["final_max_phase"] <= 1.0 + 1.0e-8:
        checks.append("phase_bounds_ok")
    if summary["min_phase_increment"] >= -1.0e-8:
        checks.append("irreversibility_ok")
    if math.isfinite(summary["final_total_energy"]):
        checks.append("finite_energy_ok")
    run_settings = summary["run_settings"]
    if run_settings["chara_length"] <= run_settings["max_chara_length"] + 1.0e-14:
        checks.append("mesh_resolution_ok")
    return checks


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if not hasattr(args, "requested_chara_length") or not hasattr(args, "max_chara_length"):
        enforce_phase_field_mesh_resolution(args)

    mesh, mesh_source = build_mesh(args)
    mesh = mesh.to(device=device, dtype=dtype)
    masks = boundary_masks(mesh.points)

    model = ZhouTensionPhaseFieldModel.from_mesh(
        mesh,
        quadrature_order=args.quadrature_order,
        E=MATERIAL["E"],
        nu=MATERIAL["nu"],
        Gc=MATERIAL["Gc"],
        l0=args.length_scale,
        k=MATERIAL["k"],
    )
    phase_laplace_row, phase_laplace_col, phase_laplace_values, phase_laplacian_scheme = build_phase_laplacian(mesh)

    displacement = torch.zeros((mesh.n_points, 2), dtype=dtype, device=device, requires_grad=True)
    phase = initial_phase_from_points(mesh.points, MATERIAL["Gc"], args.length_scale, PAPER_NUMERICS["B"], MATERIAL["k"])
    phase = phase.detach().clone().requires_grad_(True)
    history = initial_history_at_quadrature(model, PAPER_NUMERICS["B"])

    if args.uniform_steps is not None:
        displacements = uniform_displacement_schedule(args.uniform_steps, args.final_displacement)
        loading_rule = f"Uniform displacement increments to {args.final_displacement:g} mm in {args.uniform_steps} steps."
    else:
        displacements = paper_displacement_schedule(args.max_steps, args.final_displacement)
        loading_rule = "Delta u = 1e-5 mm for first 450 steps, then 1e-6 mm."
    if not displacements:
        raise RuntimeError("No load steps were requested.")

    rows: list[dict[str, float]] = []
    step_snapshots: list[dict] = []
    max_top_uy_error = 0.0
    max_top_ux_error = 0.0
    max_bottom_error = 0.0
    min_phase_increment = float("inf")

    previous_phase = phase.detach().clone()
    run_start = time.perf_counter()

    for step, prescribed in enumerate(displacements, start=1):
        step_start = time.perf_counter()
        for split_iter in range(1, args.max_stagger_iters + 1):
            u_old = apply_tension_bc(displacement, masks, prescribed, args.fix_side_x).detach()
            phi_old = phase.detach().clone()

            solve_solid_mechanics_step(
                model,
                displacement,
                phase,
                masks,
                prescribed,
                args.fix_side_x,
                args.u_iters,
                args.batch_size,
            )

            history = max_history(history, tensile_energy_at_quadrature(model, displacement.detach()))

            solve_phase_field_step(
                model,
                phase,
                history,
                previous_phase,
                phase_laplace_row,
                phase_laplace_col,
                phase_laplace_values,
            )

            du = relative_change(apply_tension_bc(displacement, masks, prescribed, args.fix_side_x).detach(), u_old)
            dphi = relative_change(phase.detach(), phi_old)
            if args.max_stagger_iters > 1 and max(du, dphi) < args.tolerance:
                break

        with torch.no_grad():
            u_solution = apply_tension_bc(displacement, masks, prescribed, args.fix_side_x)
            displacement.copy_(u_solution)
            phase_increment = float(torch.min(phase.detach() - previous_phase).cpu())
            min_phase_increment = min(min_phase_increment, phase_increment)
            previous_phase = phase.detach().clone()

            top_uy_error = torch.max(torch.abs(u_solution[masks["top"], 1] - prescribed)).item()
            top_ux_error = torch.max(torch.abs(u_solution[masks["top"], 0])).item()
            bottom_error = torch.max(torch.linalg.norm(u_solution[masks["bottom"]], dim=1)).item()
            max_top_uy_error = max(max_top_uy_error, top_uy_error)
            max_top_ux_error = max(max_top_ux_error, top_ux_error)
            max_bottom_error = max(max_bottom_error, bottom_error)

        crack = crack_metrics(
            mesh,
            phase.detach(),
            threshold=args.crack_threshold,
            band=args.crack_band,
            through_tol=args.through_tol,
        )
        should_postprocess = (
            step == 1
            or step == len(displacements)
            or step % args.postprocess_every == 0
            or bool(crack["is_through"])
        )
        if should_postprocess:
            total = float(model_energy(model, displacement.detach(), phase.detach(), args.batch_size).detach().cpu())
            dissipated = float(fracture_energy(model, phase.detach(), args.batch_size).detach().cpu())
            elastic = total
            reaction = compute_reaction(model, displacement.detach(), phase.detach(), masks["top"], args.batch_size)
        else:
            elastic = math.nan
            dissipated = math.nan
            total = math.nan
            reaction = math.nan
        step_elapsed = time.perf_counter() - step_start
        total_elapsed = time.perf_counter() - run_start

        rows.append(
            {
                "step": float(step),
                "prescribed_displacement_mm": float(prescribed),
                "reaction_y_N_per_thickness": reaction,
                "elastic_energy_Nmm": elastic,
                "fracture_energy_Nmm": dissipated,
                "total_energy_Nmm": elastic + dissipated,
                "max_phase": float(phase.detach().max().cpu()),
                "mean_phase": float(phase.detach().mean().cpu()),
                "crack_tip_x_mm": float(crack["crack_tip_x_mm"]),
                "crack_length_mm": float(crack["crack_length_mm"]),
                "step_elapsed_s": step_elapsed,
                "total_elapsed_s": total_elapsed,
                "is_developed": float(crack["is_developed"]),
                "is_through": float(crack["is_through"]),
            }
        )

        if args.save_step_plots and step % args.step_plot_every == 0:
            step_dir = Path(args.output_dir) / "steps"
            step_file = step_dir / f"step_{step:04d}.png"
            reaction_label = f"{reaction:.3e}" if math.isfinite(reaction) else "not sampled"
            title = f"step {step}, u={prescribed:.6e} mm, R_y={reaction_label}"
            save_phase_plot(mesh, phase.detach(), displacement.detach(), step_file, title_suffix=title)
            if len(step_snapshots) < args.max_overview_steps:
                disp_cpu = displacement.detach().cpu()
                step_snapshots.append(
                    {
                        "step": step,
                        "prescribed": float(prescribed),
                        "reaction": reaction,
                        "phase": phase.detach().cpu().numpy(),
                        "disp_norm": torch.linalg.norm(disp_cpu, dim=1).numpy(),
                    }
                )

        reaction_log = f"{reaction:.6e}" if math.isfinite(reaction) else "not sampled"
        print(
            f"Step {step:04d}/{len(displacements)} "
            f"u={prescribed:.6e} mm "
            f"R_y={reaction_log} "
            f"phi_max={rows[-1]['max_phase']:.4f} "
            f"tip_x={rows[-1]['crack_tip_x_mm']:.4f} "
            f"crack_length={rows[-1]['crack_length_mm']:.4f} mm "
            f"step_time={step_elapsed:.3f} s "
            f"total_time={total_elapsed:.1f} s "
            f"through={bool(crack['is_through'])}",
            flush=True,
        )

        if args.stop_on_through and crack["is_through"]:
            print(f"Crack reached the right boundary at step {step}, u={prescribed:.6e} mm.")
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "load_displacement.csv"
    curve_file = output_dir / "load_displacement_curve.png"
    plot_file = output_dir / "final_phase_field.png"
    overview_file = output_dir / "loading_steps_overview.png"
    summary_file = output_dir / "validation_summary.json"

    write_csv(rows, csv_file)
    save_load_curve(rows, curve_file)
    final_u = apply_tension_bc(displacement, masks, rows[-1]["prescribed_displacement_mm"], args.fix_side_x).detach()
    final_phase = phase.detach()
    save_phase_plot(mesh, final_phase, final_u, plot_file)
    if args.save_step_plots:
        save_loading_overview(mesh, step_snapshots, overview_file)

    bounds_min = mesh.points.detach().min(dim=0).values.cpu().tolist()
    bounds_max = mesh.points.detach().max(dim=0).values.cpu().tolist()
    summary = {
        "reference": {
            "paper": "Zhou S, Rabczuk T, Zhuang X. Advances in Engineering Software, 2018, 122:31-49.",
            "doi": "10.1016/j.advengsoft.2018.05.005",
            "method": "AT2 phase-field fracture with spectral tensile/compressive split, weak-form mechanics, strong-form phase solve.",
        },
        "mesh_source": mesh_source,
        "device": str(device),
        "dtype": args.dtype,
        "geometry_mm": GEOMETRY,
        "material": MATERIAL,
        "paper_numerics": PAPER_NUMERICS,
        "run_settings": {
            "length_scale": args.length_scale,
            "chara_length": args.chara_length,
            "requested_chara_length": args.requested_chara_length,
            "max_chara_length": args.max_chara_length,
            "mesh_resolution_rule": "chara_length <= 0.5 * length_scale",
            "mesh_resolution_clamped": args.chara_length < args.requested_chara_length,
            "element_type": args.element_type,
            "quadrature_order": args.quadrature_order,
            "solution_scheme": "mechanics_first_split_with_strong_phase",
            "mechanics_form": "weak_energy_minimization",
            "phase_form": "strong_linear_system",
            "phase_laplacian_scheme": phase_laplacian_scheme,
            "split_iterations_per_step": args.max_stagger_iters,
            "fix_side_x": args.fix_side_x,
            "loading_steps_requested": len(displacements),
            "loading_steps_run": len(rows),
            "final_displacement": rows[-1]["prescribed_displacement_mm"],
            "loading_rule": loading_rule,
            "paper_loading_rule": "Delta u = 1e-5 mm for first 450 steps, then 1e-6 mm.",
            "uses_uniform_loading": args.uniform_steps is not None,
            "crack_threshold": args.crack_threshold,
            "crack_band": args.crack_band,
            "through_tol": args.through_tol,
            "stop_on_through": args.stop_on_through,
            "save_step_plots": args.save_step_plots,
            "step_plot_every": args.step_plot_every,
            "postprocess_every": args.postprocess_every,
        },
        "mesh": {
            "n_points": int(mesh.n_points),
            "n_elements": int(mesh.n_elements),
            "x_min": float(bounds_min[0]),
            "x_max": float(bounds_max[0]),
            "y_min": float(bounds_min[1]),
            "y_max": float(bounds_max[1]),
        },
        "max_top_uy_error": max_top_uy_error,
        "max_top_ux_error": max_top_ux_error,
        "max_bottom_displacement_error": max_bottom_error,
        "min_phase_increment": min_phase_increment,
        "initial_max_phase": float(initial_phase_from_points(mesh.points, MATERIAL["Gc"], args.length_scale, PAPER_NUMERICS["B"], MATERIAL["k"]).max().detach().cpu()),
        "final_max_phase": float(final_phase.max().cpu()),
        "final_mean_phase": float(final_phase.mean().cpu()),
        "final_elastic_energy": rows[-1]["elastic_energy_Nmm"],
        "final_fracture_energy": rows[-1]["fracture_energy_Nmm"],
        "final_total_energy": rows[-1]["total_energy_Nmm"],
        "final_crack_tip_x_mm": rows[-1]["crack_tip_x_mm"],
        "final_crack_length_mm": rows[-1]["crack_length_mm"],
        "final_is_developed": bool(rows[-1]["is_developed"]),
        "final_is_through": bool(rows[-1]["is_through"]),
        "csv_file": str(csv_file),
        "curve_file": str(curve_file),
        "plot_file": str(plot_file),
        "step_plot_dir": str(output_dir / "steps") if args.save_step_plots else None,
        "overview_file": str(overview_file) if args.save_step_plots else None,
    }
    summary["checks_passed"] = validate_summary(summary)
    summary["validation_passed"] = len(summary["checks_passed"]) == 9

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Validation summary: {summary_file}")
    print(f"Final plot: {plot_file}")
    print(f"Load curve: {curve_file}")
    if args.save_step_plots:
        print(f"Step plots: {output_dir / 'steps'}")
        print(f"Loading overview: {overview_file}")
    print(f"Load-displacement data: {csv_file}")
    print(f"Validation passed: {summary['validation_passed']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mesh-file", default=None)
    parser.add_argument("--element-type", default="quad", choices=["quad", "tri"])
    parser.add_argument(
        "--chara-length",
        type=float,
        default=PAPER_NUMERICS["max_h_over_l0"] * PAPER_NUMERICS["l0_large"],
        help="Characteristic mesh size. Values above 0.5 * --length-scale are reduced automatically.",
    )
    parser.add_argument("--paper-mesh", action="store_true", help="Use h=3.96e-3 mm from the paper.")
    parser.add_argument("--length-scale", type=float, default=PAPER_NUMERICS["l0_large"])
    parser.add_argument("--small-length-scale", action="store_true", help="Use l0=7.5e-3 mm from the paper.")
    parser.add_argument("--quadrature-order", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="Run the paper load schedule up to --final-displacement.")
    parser.add_argument("--final-displacement", type=float, default=5.3e-3)
    parser.add_argument("--uniform-steps", type=int, default=None, help="Use uniformly spaced displacement steps up to --final-displacement.")
    parser.add_argument(
        "--max-stagger-iters",
        type=int,
        default=1,
        help="Number of mechanics-first split cycles per load step. The default 1 solves mechanics once, then phase once.",
    )
    parser.add_argument("--u-iters", type=int, default=20)
    parser.add_argument(
        "--phase-iters",
        type=int,
        default=20,
        help="Deprecated: retained for CLI compatibility. The phase field is solved by a strong-form sparse linear system.",
    )
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--tolerance", type=float, default=PAPER_NUMERICS["tolerance"])
    parser.add_argument("--free-side-x", action="store_true", help="Do not constrain horizontal displacement on side edges.")
    parser.add_argument("--no-step-plots", action="store_true", help="Do not save one image per loading step.")
    parser.add_argument("--step-plot-every", type=int, default=5, help="Save one step image every N loading steps.")
    parser.add_argument(
        "--postprocess-every",
        type=int,
        default=5,
        help="Compute reaction force and energy every N loading steps. The first, final, and through-crack steps are always sampled.",
    )
    parser.add_argument("--max-overview-steps", type=int, default=12, help="Maximum number of steps shown in the overview image.")
    parser.add_argument(
        "--stop-on-through",
        dest="stop_on_through",
        action="store_true",
        default=True,
        help="Stop when the monitored crack reaches the right boundary. This is enabled by default.",
    )
    parser.add_argument(
        "--continue-after-through",
        dest="stop_on_through",
        action="store_false",
        help="Continue loading after the monitored crack reaches the right boundary.",
    )
    parser.add_argument("--crack-threshold", type=float, default=0.6, help="Phase-field threshold used to monitor crack advance.")
    parser.add_argument("--crack-band", type=float, default=0.08, help="Half-width around y=0.5 mm used to monitor horizontal crack advance.")
    parser.add_argument("--through-tol", type=float, default=0.04, help="Distance from right edge treated as through-crack.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    args = parser.parse_args()

    if args.paper_mesh:
        args.chara_length = PAPER_NUMERICS["paper_h"]
    if args.small_length_scale:
        args.length_scale = PAPER_NUMERICS["l0_small"]
    if args.length_scale <= 0.0:
        raise ValueError("--length-scale must be positive.")
    if args.chara_length <= 0.0:
        raise ValueError("--chara-length must be positive.")
    enforce_phase_field_mesh_resolution(args)
    if args.full:
        args.max_steps = None
    args.fix_side_x = not args.free_side_x
    args.save_step_plots = not args.no_step_plots
    args.step_plot_every = max(1, args.step_plot_every)
    args.postprocess_every = max(1, args.postprocess_every)
    args.max_overview_steps = max(1, args.max_overview_steps)
    args.max_stagger_iters = max(1, args.max_stagger_iters)
    return args


def main() -> None:
    args = parse_args()
    summary = run(args)
    if not summary["validation_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
