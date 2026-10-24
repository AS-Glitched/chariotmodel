"""Stage report builder for the W1 -> W7 improvement programme.

Every stage lands the same artefacts:

``runs/<stage_id>/summary.md``
    A human-readable report: the protocol and the split boundaries, the
    content hashes, the model version, the 2022-validation (selection)
    metrics, the 2023-test (report-only) metrics, the baseline table, the
    gate verdicts and the claims that are allowed and forbidden.

``runs/<stage_id>/comparison.json``
    The same content in machine-readable form.

The builder is deliberately dumb: it never computes a metric and never
changes a model.  It only formats what the caller passes in, so a report can
never drift away from the runs it describes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = ["build_stage_report", "render_markdown"]


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a metric value for the markdown tables."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "inf"
        return f"{value:,.{digits}f}"
    return str(value)


def _metric_row(name: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "model": name,
        "mae_w": metrics.get("mae_mean"),
        "rmse_w": metrics.get("rmse_mean"),
        "bias_w": metrics.get("bias_mean"),
        "coverage": metrics.get("coverage_mean"),
        "energy_error_wh": metrics.get("energy_error_wh_mean"),
        "survival": metrics.get("survival_rate"),
        "min_survival": metrics.get("min_survival_rate"),
        "n_windows": metrics.get("n_windows"),
    }


def build_stage_report(
    *,
    stage_id: str,
    title: str,
    out_dir: str | Path,
    protocol: dict[str, Any],
    data: dict[str, Any],
    model: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    gates: list[dict[str, Any]],
    claims: dict[str, list[str]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``summary.md`` + ``comparison.json`` for one stage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, Any] = {
        "stage_id": stage_id,
        "title": title,
        "protocol": protocol,
        "data": data,
        "model": model,
        "metrics": metrics,
        "baselines": baselines,
        "gates": gates,
        "claims": claims,
        "notes": list(notes or []),
    }
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=False, default=str) + "\n"
    )
    (out / "summary.md").write_text(render_markdown(comparison))
    return comparison


def render_markdown(c: dict[str, Any]) -> str:
    """Render the stage report as markdown."""
    lines: list[str] = []
    add = lines.append

    add(f"# {c['stage_id']} — {c['title']}")
    add("")

    protocol = c.get("protocol", {})
    add("## Protocol")
    add("")
    for key, value in protocol.items():
        add(f"- **{key}**: {value}")
    add("")

    data = c.get("data", {})
    if data:
        add("### Data and content hashes")
        add("")
        hashes = data.get("content_hashes", {})
        if hashes:
            add("| slice | content hash |")
            add("| --- | --- |")
            for name, digest in hashes.items():
                add(f"| {name} | `{digest}` |")
            add("")
        boundaries = data.get("split_boundaries", {})
        if boundaries:
            add("| split | first row | last row | rows |")
            add("| --- | --- | --- | --- |")
            for split, b in boundaries.items():
                if not isinstance(b, dict):
                    continue
                # The data manifest nests the boundaries per slice
                # ({"train_2022": {"train": {...}}}); the dataset manifest
                # is flat ({"train": {...}}).
                entries = b if "min" in b else {
                    k: v for k, v in b.items() if isinstance(v, dict) and "min" in v
                }
                if "min" in b:
                    entries = {split: b}
                for name, bounds in entries.items():
                    add(
                        f"| {name} | {bounds.get('min')} | {bounds.get('max')} "
                        f"| {_fmt(bounds.get('count'))} |"
                    )
            add("")

    model = c.get("model", {})
    if model:
        add("### Model")
        add("")
        for key, value in model.items():
            add(f"- **{key}**: {value}")
        add("")

    metrics = c.get("metrics", {})
    if metrics:
        add("## Metrics")
        add("")
        for split, split_metrics in metrics.items():
            add(f"### {split}")
            add("")
            add("| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            add(
                "| ops | {mae} | {rmse} | {bias} | {cov} | {energy} | {surv} | {mins} | {n} |".format(
                    mae=_fmt(split_metrics.get("mae_mean")),
                    rmse=_fmt(split_metrics.get("rmse_mean")),
                    bias=_fmt(split_metrics.get("bias_mean")),
                    cov=_fmt(split_metrics.get("coverage_mean")),
                    energy=_fmt(split_metrics.get("energy_error_wh_mean")),
                    surv=_fmt(split_metrics.get("survival_rate")),
                    mins=_fmt(split_metrics.get("min_survival_rate")),
                    n=_fmt(split_metrics.get("n_windows")),
                )
            )
            add("")

    baselines = c.get("baselines", {})
    if baselines:
        add("## Baselines")
        add("")
        add("| baseline | MAE (W) | RMSE (W) | bias (W) | energy error (Wh) | windows |")
        add("| --- | --- | --- | --- | --- | --- |")
        for name, b in baselines.items():
            add(
                "| {name} | {mae} | {rmse} | {bias} | {energy} | {n} |".format(
                    name=name,
                    mae=_fmt(b.get("mae_mean")),
                    rmse=_fmt(b.get("rmse_mean")),
                    bias=_fmt(b.get("bias_mean")),
                    energy=_fmt(b.get("energy_error_wh_mean")),
                    n=_fmt(b.get("n_windows")),
                )
            )
        add("")

    gates = c.get("gates", [])
    if gates:
        add("## Gates")
        add("")
        add("| gate | target | observed | verdict |")
        add("| --- | --- | --- | --- |")
        for gate in gates:
            verdict = "PASS" if gate.get("pass") else "FAIL"
            add(
                "| {name} | {target} | {observed} | {verdict} |".format(
                    name=gate.get("name", ""),
                    target=gate.get("target", ""),
                    observed=gate.get("observed", ""),
                    verdict=verdict,
                )
            )
            if gate.get("note"):
                add("")
                add(f"> {gate['note']}")
                add("")
        add("")

    claims = c.get("claims", {})
    if claims:
        add("## Claims")
        add("")
        add("**Allowed**")
        add("")
        for item in claims.get("allowed", []):
            add(f"- {item}")
        add("")
        add("**Forbidden**")
        add("")
        for item in claims.get("forbidden", []):
            add(f"- {item}")
        add("")

    notes = c.get("notes", [])
    if notes:
        add("## Notes")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")

    return "\n".join(lines)
