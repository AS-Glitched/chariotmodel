# CHARIOT lab prototype

Physics-governed advisory prototype for one solar + storage site.

```text
synthetic telemetry
  → Admissibility
  → sealed extract + SHA-256
  → time-ordered split
  → baselines
  → HC-BSSWM v1 (clear-sky Bayesian state-space world model)
  → Reality Tribunal
  → Grid Engineer / Researcher / Investor-Planner advisories
```

Outputs are advisory only. If physics survival is too low, the system abstains.

HC-BSSWM v1 runs on CPU with NumPy. It does not need a GPU or PyMC. That keeps it stable on a 16 GB workstation (Xeon W-2223 class).

## Open in VS Code

1. Unzip this folder.
2. File → Open Folder → select `chariot-vscode`.
3. Terminal:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_prototype.py
python -m pytest tests -q
```

Or use Run and Debug → **CHARIOT prototype** (install the Python extension).

## Outputs

`runs/latest/`

| File | Mode |
|------|------|
| `manifest.json` | versions, hash, survival rate, baselines |
| `grid_engineer.json` | operational P10/P50/P90, SOC risk |
| `researcher.json` | lineage, diagnostics notes, rejections |
| `investor_planner.json` | planning yield band, not a financial guarantee |
| `sealed_extract.parquet` | frozen training table |

## What this is not

- Not multi-site hierarchical pooling yet (same model name; single-site v1).
- Not inverter control.
- Not a price or bidding engine.
- PyMC NUTS can replace `chariot/hcbsswm.py` later without changing the Tribunal or the three modes.
