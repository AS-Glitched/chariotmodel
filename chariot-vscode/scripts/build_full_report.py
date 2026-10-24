#!/usr/bin/env python3
"""Assemble the COMPLETE CHARIOT report into a single plain-text file.

The output (``docs/CHARIOT_FULL_REPORT.txt``) contains, in one file:

* the executive summary and the key results;
* the architecture (module map, contracts, the data flow);
* every formula the system uses;
* the logic (the algorithms and every decision rule);
* every test and what it asserts;
* every failure that was hit and how it was resolved;
* the measured results of the v3 run and of the ops-vs-PyMC comparison;
* the tools, the dependencies, the environment adaptations and the commands;
* **the complete source code** of every module, script and test;
* the run artefacts (manifests, summaries, the full comparison JSON).

Everything is read from the repository — the file is a *generated* artefact and
is regenerated with::

    python scripts/build_full_report.py
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "docs" / "CHARIOT_FULL_REPORT.txt"

RULE = "=" * 78
THIN = "-" * 78

SOURCE_FILES: list[tuple[str, str]] = [
    ("chariot/__init__.py", "the package marker and the v3 identity"),
    ("chariot/config.py", "v3 identity, versioned policies, horizons, SiteContext"),
    ("chariot/hashing.py", "canonical JSON + SHA-256 lineage helpers"),
    ("chariot/store.py", "SQLite lineage/audit store (stdlib only)"),
    ("chariot/site_master.py", "site registry + effective-context resolution"),
    ("chariot/admissibility.py", "hard/soft fail-closed physics rule engine"),
    ("chariot/river_flow.py", "content-addressed slices + replay certificates"),
    ("chariot/calibration.py", "versioned, scoped, audited calibration"),
    ("chariot/capsule.py", "immutable capsule sealing/verification"),
    ("chariot/training_extract.py", "materialised dataset + clear-sky gate + 60/20/20 split"),
    ("chariot/physics.py", "bus-balance solver (the Tribunal's energy accounting)"),
    ("chariot/model.py", "the OPERATIONAL NumPy clearness state-space model"),
    ("chariot/pymc_clearness.py", "the OFFLINE PyMC reference model + adapter"),
    ("chariot/residual.py", "Residual Proposal Chamber (ridge, capped, versioned)"),
    ("chariot/genesis.py", "Scenario Genesis Engine"),
    ("chariot/tribunal.py", "Reality Tribunal + Rejection Echo Memory"),
    ("chariot/governor.py", "Residual Stability Governor (sole runtime authority for w)"),
    ("chariot/feedback.py", "Dual Feedback Cortex"),
    ("chariot/confidence.py", "Physics-Survival Confidence Engine"),
    ("chariot/advisory.py", "role modes, quantiles, events, abstention"),
    ("chariot/metrics.py", "evaluation metrics + baseline scoreboard"),
    ("chariot/pipeline.py", "the end-to-end v3 spine"),
    ("chariot/api.py", "FastAPI surface (50 operations / 46 paths, advisory-only)"),
    ("scripts/prepare_dataset.py", "materialise the Parquet slice + the 60/20/20 split"),
    ("scripts/run_chariot3.py", "run the full operational spine"),
    ("scripts/diagnose_model.py", "clearness-channel diagnostics"),
    ("scripts/run_pymc_reference.py", "the ops-vs-PyMC reference comparison harness"),
    ("scripts/build_full_report.py", "this generator"),
    ("tests/conftest.py", "the synthetic reference site + the pytest markers"),
    ("tests/test_spine.py", "the v3 invariants"),
    ("tests/test_api.py", "the HTTP surface"),
    ("tests/test_pymc_reference.py", "the offline reference path"),
]

ARTEFACTS: list[tuple[str, str]] = [
    ("runs/chariot3_alice_springs_2022/manifest.json", "the v3 spine manifest (every layer)"),
    ("runs/chariot3_alice_springs_2022/summary.json", "the v3 metrics + baselines"),
    ("runs/chariot3_alice_springs_2022/windows.json", "the v3 per-window rows"),
    ("runs/pymc_reference/summary.md", "the ops-vs-PyMC side-by-side table"),
    ("runs/pymc_reference/comparison.json", "the complete machine-readable comparison"),
]

EXTRA_FILES: list[tuple[str, str]] = [
    ("../.github/workflows/ci.yml", "the CI matrix (Python 3.10/3.11/3.12)"),
    ("requirements.txt", "the operational dependencies"),
    ("requirements-pymc.txt", "the optional PyMC extra"),
    ("README.md", "the README"),
    ("docs/ARCHITECTURE.md", "the architecture document"),
]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - defensive
        return ""


def _banner(title: str, subtitle: str = "") -> str:
    out = [RULE, title, RULE]
    if subtitle:
        out.append(subtitle)
    return "\n".join(out)


def _file_block(path: Path, title: str, subtitle: str = "") -> str:
    if not path.exists():
        return f"{THIN}\n[MISSING] {path}\n{THIN}\n"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    head = _banner(title, subtitle)
    return (
        f"{head}\n"
        f"file   : {path.relative_to(REPO) if path.is_relative_to(REPO) else path}\n"
        f"lines  : {lines}\n"
        f"bytes  : {len(text.encode('utf-8'))}\n"
        f"{THIN}\n{text}\n{THIN}\n"
    )


def build() -> str:
    parts: list[str] = []
    commit = _git("rev-parse", "HEAD")[:12]
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---------------------------------------------------------------- header
    parts.append(
        _banner(
            "CHARIOT v3 + THE OFFLINE PyMC REFERENCE — COMPLETE ENGINEERING REPORT",
            "Every formula, every rule, every test, every result and the complete "
            "source code, in one file.",
        )
    )
    parts.append(
        "\n".join(
            [
                f"repository   : https://github.com/AS-Glitched/chariotmodel",
                f"branch       : {branch}",
                f"commit       : {commit}",
                f"pull request : #1 (open)",
                f"generated    : {now}",
                f"generator    : scripts/build_full_report.py",
                "",
                "CONTENTS",
                "  PART 0  Executive summary and the key numbers",
                "  PART 1  Architecture: the layer map, the contracts, the data flow",
                "  PART 2  Every formula the system uses",
                "  PART 3  The logic: every algorithm and every decision rule",
                "  PART 4  Every test and what it asserts",
                "  PART 5  Every failure hit and how it was resolved",
                "  PART 6  The measured results",
                "  PART 7  Tools, dependencies, environment, commands",
                "  PART 8  The complete source code, file by file",
                "  PART 9  The run artefacts",
                "  PART 10 The commit history",
                "",
                "PARTS 0-7 are the engineering report (docs/REPORT.md) verbatim;",
                "PART 8 is every module, script and test in full;",
                "PART 9 is every run artefact that backs the numbers quoted above.",
                "",
            ]
        )
    )

    # ------------------------------------------- PART 0-7 (the written report)
    report = ROOT / "docs" / "REPORT.md"
    parts.append(
        _banner(
            "PARTS 0-7 — THE ENGINEERING REPORT",
            "docs/REPORT.md, reproduced verbatim.",
        )
    )
    if report.exists():
        parts.append(report.read_text(encoding="utf-8"))
    else:  # pragma: no cover - defensive
        parts.append("[MISSING] docs/REPORT.md\n")

    # ------------------------------------------------------- PART 8 (source)
    parts.append(_banner("PART 8 — THE COMPLETE SOURCE CODE", "Every file, in full."))
    for rel, purpose in SOURCE_FILES:
        parts.append(_file_block(ROOT / rel, f"SOURCE: {rel}", purpose))
    for rel, purpose in EXTRA_FILES:
        parts.append(_file_block((ROOT / rel).resolve(), f"FILE: {rel}", purpose))

    # --------------------------------------------------- PART 9 (artefacts)
    parts.append(
        _banner(
            "PART 9 — THE RUN ARTEFACTS",
            "The files that back every number quoted in Parts 0-7.",
        )
    )
    for rel, purpose in ARTEFACTS:
        parts.append(_file_block(ROOT / rel, f"ARTEFACT: {rel}", purpose))

    parts.append(_history_block())
    parts.append(
        _banner("END OF REPORT", f"commit {commit} — {branch} — {now}")
    )
    return "\n".join(parts)


def _history_block() -> str:
    """The commit history that produced the state described above."""
    log = _git("log", "--oneline", "--decorate", "-20")
    stat = _git("show", "--stat", "--oneline", "HEAD")
    body = [
        _banner("PART 10 — THE COMMIT HISTORY", "How the repository got here."),
        log or "(no git history available)",
        "",
        THIN,
        "The most recent commit in detail:",
        stat or "",
    ]
    return "\n".join(body)


def main() -> int:
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size:,} bytes, {text.count(chr(10)) + 1:,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
