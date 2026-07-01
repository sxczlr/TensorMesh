# Zhou et al. 2018 notched tension benchmark

This TensorMesh example implements the 2D single-edge-notched square plate
subjected to quasi-static tension from Zhou, Rabczuk and Zhuang,
*Advances in Engineering Software* 122, 31-49 (2018).

Matched method and parameters:

- AT2 crack surface density: `phi^2/(2 l0) + l0/2 |grad phi|^2`
- spectral tensile/compressive strain-energy split from Miehe et al.
- degradation `[(1-k)(1-phi)^2 + k]` with `k = 1e-9`
- history field `H = max psi_plus` and initial notch history field with `B = 1e6`
- staggered displacement/history/phase-field solution
- geometry: `1 mm x 1 mm` square plate, notch from `(0, 0.5)` to `(0.5, 0.5)` mm
- material: `E = 210 GPa`, `nu = 0.3`, `Gc = 2700 J/m^2`
- length scales available from the paper: `l0 = 1.5e-2 mm` (default) and `7.5e-3 mm`
- generated mesh size is constrained by `h <= 0.5 l0` so the diffuse crack band is resolved
- paper loading rule: `Delta u = 1e-5 mm` for the first 450 steps, then `1e-6 mm`
- the run stops by default once the monitored crack reaches the right boundary
- reaction force and energy are sampled every 5 steps by default to reduce post-processing cost
- Matplotlib step plots are saved every 5 steps by default

Run a quick validation in the PINN environment:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py
```

The default `--device auto` uses CUDA when available. To require GPU execution:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --device cuda
```

Use the smaller paper length scale:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --small-length-scale
```

If `--chara-length` is set larger than half of `--length-scale`, the script
automatically reduces it to `0.5 * l0`. For the default `l0 = 1.5e-2 mm`,
the generated mesh size is therefore `h = 7.5e-3 mm`.

Use the paper mesh size. This is much heavier because the paper used about
64516 Q4 elements:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --paper-mesh --max-steps 1
```

Outputs are written under `results/` by default:

- `validation_summary.json`
- `load_displacement.csv`
- `final_phase_field.png`
- `steps/step_XXXX.png` for every saved loading step
- `loading_steps_overview.png` with the saved loading sequence

Show the first 10 tensile loading steps:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --max-steps 10 --output-dir .\results_10steps
```

For long runs, save one image every 10 loading steps:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --full --step-plot-every 10 --postprocess-every 10 --output-dir .\results_full
```

Load directly to `6e-3 mm` with uniformly spaced display steps:

```powershell
& 'D:\APP\PINN\Scripts\python.exe' .\phase_field_zhou_tension.py --final-displacement 6e-3 --uniform-steps 120 --step-plot-every 10 --postprocess-every 10 --output-dir .\results_u6e-3
```

To continue loading after a through-crack, add `--continue-after-through`.
