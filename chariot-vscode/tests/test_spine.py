import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.admissibility import evaluate_row
from chariot.extract import content_hash, time_split
from chariot.pipeline import run
from chariot.synthetic import make_site_frame
import pandas as pd


def test_hash_stable():
    frame = make_site_frame(days=2, seed=1)
    assert content_hash(frame) == content_hash(frame)


def test_negative_solar_rejected():
    frame = make_site_frame(days=1, seed=1)
    row = frame.iloc[10].copy()
    row["solar_w"] = -5
    report = evaluate_row(row)
    assert report["decision"] == "REJECTED"
    assert "SOLAR_NON_NEGATIVE" in report["hard_failures"]


def test_time_split_no_future():
    frame = make_site_frame(days=2, seed=1)
    train, test = time_split(frame)
    assert train["timestamp_utc"].max() <= test["timestamp_utc"].min()


def test_pipeline_manifest():
    manifest = run(out_dir=ROOT / "runs" / "test", days=4, n_scenarios=40, seed=3)
    assert manifest["lineage"]["model_name"] == "chariot_hcbsswm"
    assert 0.0 <= manifest["survival_rate"] <= 1.0
    assert manifest["status"] in {"OK", "REDUCED_CONFIDENCE", "ABSTAIN"}
    assert Path(ROOT / "runs" / "test" / "grid_engineer.json").exists()
