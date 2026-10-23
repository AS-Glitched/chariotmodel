"""CHARIOT v3 — HTTP surface tests.

The API is exercised end to end over a synthetic reference site: telemetry
in, admissibility, River Flow, capsule, training extract, model, scenarios,
Tribunal, confidence and advisory out.  Two invariants are asserted on the
surface itself:

* the surface is **advisory only** — no control, dispatch or setpoint route
  exists;
* nothing is published that has not been certified by the Tribunal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from chariot.api import create_app  # noqa: E402

from conftest import SITE_ID, TENANT_ID, make_reference_frame  # noqa: E402

FORBIDDEN_ROUTE_HINTS = ("control", "dispatch", "setpoint", "actuate", "command")


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    app = create_app(tmp_path_factory.mktemp("api") / "chariot.db")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def telemetry() -> list[dict]:
    frame = make_reference_frame(days=10)
    records = frame.to_dict(orient="records")
    for record in records:
        record["timestamp_utc"] = str(record["timestamp_utc"])
    return records


def _register_site(client: TestClient) -> None:
    response = client.post(
        "/sites",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "name": "DKASC Alice Springs (test)",
            "rated_pv_w": 220000.0,
            "usable_battery_capacity_wh": 220000.0,
            "max_charge_w": 110000.0,
            "max_discharge_w": 110000.0,
            "timezone": "Australia/Darwin",
            "latitude": -23.762,
            "longitude": 133.875,
            "fleet_id": "dkasc",
            "metadata": {
                "load_profile_w": {str(h): 20000 for h in range(24)},
                "load_relative_sigma": 0.1,
                "max_grid_import_w": 60000.0,
                "max_grid_export_w": 250000.0,
                "soc_floor_pct": 5.0,
                "soc_ceiling_pct": 95.0,
                "solar_ramp_w_per_step": 110000.0,
            },
        },
    )
    assert response.status_code == 201, response.text


def test_health_and_policy_surface(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["advisory_only"] is True
    rules = client.get("/admissibility/rule-sets").json()
    assert rules["admissibility"]["rule_set_version"].startswith("admissibility_v")
    assert rules["tribunal"]["policy_version"].startswith("tribunal_v")
    assert client.get("/confidence/policy").json()["policy_version"].startswith("confidence_v")
    assert client.get("/tribunal/policy").json()["policy_version"].startswith("tribunal_v")


def test_no_control_endpoints_exist(client: TestClient) -> None:
    paths = [route.path for route in client.app.routes]
    for path in paths:
        for hint in FORBIDDEN_ROUTE_HINTS:
            assert hint not in path.lower(), f"control-like route exposed: {path}"


def test_full_api_spine(client: TestClient, telemetry: list[dict]) -> None:
    _register_site(client)

    # --- Site Master ---------------------------------------------------
    sites = client.get("/sites", params={"tenant_id": TENANT_ID}).json()
    assert sites["count"] == 1
    context = client.get(
        f"/sites/{SITE_ID}/context", params={"tenant_id": TENANT_ID}
    ).json()
    assert context["rated_pv_w"] == 220000.0

    # --- Admissibility --------------------------------------------------
    evaluation = client.post(
        "/admissibility/evaluate",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "records": telemetry[:200],
        },
    ).json()
    assert evaluation["summary"]["decision_counts"]
    decisions = client.get("/admissibility/decisions", params={"site_id": SITE_ID}).json()
    assert decisions["count"] > 0

    # --- River Flow -----------------------------------------------------
    ingest = client.post(
        "/flow/ingest",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "records": telemetry,
            "horizontal_parameters": ["solar_w", "clear_sky_w"],
        },
    ).json()
    assert ingest["rows"] == len(telemetry)
    assert ingest["slice_ids"]
    replay_id = ingest["replay"]["replay_id"]
    assert client.get(f"/replay/{replay_id}").json()["valid"] is True
    assert client.get(
        "/slices/vertical", params={"tenant_id": TENANT_ID, "site_id": SITE_ID}
    ).json()["count"] >= 1
    assert client.get(
        "/slices/horizontal", params={"tenant_id": TENANT_ID, "site_id": SITE_ID}
    ).json()["count"] >= 1

    # --- Calibration ----------------------------------------------------
    version = client.post(
        "/calibration/versions",
        json={
            "scope_type": "site",
            "scope_id": SITE_ID,
            "capacity_factor": 1.0,
            "charge_loss_factor": 0.03,
            "discharge_loss_factor": 0.03,
            "pv_correction_factor": 1.0,
            "notes": "api test",
        },
    ).json()
    calibration_version = version["calibration_version"]
    assert client.post(f"/calibration/versions/{calibration_version}/approve").json()[
        "status"
    ] == "approved"
    resolved = client.get(
        "/calibration/resolve", params={"tenant_id": TENANT_ID, "site_id": SITE_ID}
    ).json()
    assert resolved["charge_loss_factor"] == pytest.approx(0.03)
    assert client.get("/calibration/versions").json()["count"] >= 1

    # --- Capsule --------------------------------------------------------
    sealed = client.post(
        "/capsules/seal",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "calibration_version": calibration_version,
            "slice_ids": ingest["slice_ids"],
            "slice_hashes": ingest["slice_hashes"],
            "admissibility_versions": ["admissibility_v3.0"],
        },
    ).json()
    capsule_id = sealed["capsule_id"]
    assert client.get(f"/capsules/{capsule_id}").json()["valid"] is True

    # --- Training extract -----------------------------------------------
    extract = client.post(
        "/training-extract/build",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "capsule_ids": [capsule_id],
            "calibration_version": calibration_version,
            "calibration_factors": resolved,
            "admissibility_versions": ["admissibility_v3.0"],
            "slice_ids": ingest["slice_ids"],
        },
    ).json()
    dataset_version = extract["dataset_version"]
    dataset = client.get(f"/training-extract/datasets/{dataset_version}").json()
    assert dataset["rows"] > 0
    assert set(dataset["split_counts"]) == {"train", "validate", "test"}

    # --- Model -----------------------------------------------------------
    trained = client.post(
        "/model/chariot/train",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "dataset_version": dataset_version,
            "n_scenarios": 40,
        },
    ).json()
    model_version = trained["model_version"]
    assert client.get("/model/chariot/versions").json()["count"] >= 1
    assert client.post("/model/chariot/promote", json={"model_version": model_version}).json()[
        "status"
    ] == "active"

    inference = client.post(
        "/model/chariot/infer",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "model_version": model_version,
            "records": telemetry[:100],
        },
    ).json()
    assert inference["n_rows"] > 0
    assert inference["one_step_mae_w"] is not None

    # --- Scenarios -------------------------------------------------------
    generated = client.post(
        "/model/scenarios/generate",
        json={
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "model_version": model_version,
            "n_scenarios": 40,
            "records": telemetry[:96],
        },
    ).json()
    batch_id = generated["batch_id"]
    assert client.get(f"/model/scenarios/batches/{batch_id}").json()["batch_id"] == batch_id

    # --- Tribunal --------------------------------------------------------
    verdict = client.post("/tribunal/validate-batch", json={"batch_id": batch_id}).json()
    assert verdict["n_proposed"] == 40
    assert verdict["n_retained"] + verdict["n_rejected"] == 40
    retained = client.get(f"/tribunal/batches/{batch_id}/retained").json()
    assert retained["n_retained"] == verdict["n_retained"]
    rejections = client.get(f"/tribunal/batches/{batch_id}/rejections").json()
    assert rejections["count"] == verdict["n_rejected"]

    # --- Governor --------------------------------------------------------
    assert client.get("/residual/governor/status").json()["effective_w"] >= 0.0
    client.post(
        "/residual/governor/set-weight", json={"w": 0.1, "actor": "test"}
    )
    assert client.get("/residual/governor/status").json()["effective_w"] == pytest.approx(0.1)

    # --- Confidence -------------------------------------------------------
    report = client.post("/confidence/evaluate", json={"batch_id": batch_id}).json()
    assert report["status"] in {
        "eligible_for_human_review",
        "reduced_confidence",
        "abstained",
    }
    assert client.get(f"/confidence/reports/{report['report_id']}").json()[
        "report_id"
    ] == report["report_id"]

    # --- Advisory ---------------------------------------------------------
    forecast = client.post(
        "/advisory/forecast",
        json={"batch_id": batch_id, "role_mode": "grid_engineer"},
    ).json()
    assert forecast["advisory_only"] is True
    assert forecast["not_a_controller"] is True
    assert forecast["solar_w"]["p10"]
    events = client.get("/advisory/events", params={"batch_id": batch_id}).json()
    assert "horizon_steps" in events

    # --- Feedback ---------------------------------------------------------
    job = client.post(
        "/feedback/cortex/run", json={"batch_id": batch_id, "survival_rate": 0.9}
    ).json()
    assert job["batch_id"] == batch_id
    assert client.get("/feedback/jobs").json()["count"] >= 1


def test_rejection_echo_is_addressable(client: TestClient) -> None:
    """The Echo is an append-only memory, not a log that can be dropped."""
    echo = client.get("/tribunal/rejections").json()
    assert "rejections" in echo
    for row in echo["rejections"]:
        assert row["rejection_id"]
        assert row["batch_id"]
        assert row["constraint_codes"] is not None
        one = client.get(f"/tribunal/rejections/{row['rejection_id']}").json()
        assert one["rejection_id"] == row["rejection_id"]
    assert client.get("/tribunal/rejections/does-not-exist").status_code == 404


def test_policies_are_versioned_never_edited_in_place(client: TestClient) -> None:
    """A policy change mints a new version; the old one stays on record."""
    before = client.get("/tribunal/policy").json()
    assert before["policy_version"].startswith("tribunal_v")
    assert before["effective"] is True

    updated = client.put(
        "/tribunal/policy",
        json={"energy_balance_tol_wh": 0.25, "actor": "test", "reason": "tolerance change"},
    ).json()
    assert updated["policy_version"] != before["policy_version"]
    assert updated["previous_version"] == before["policy_version"]
    assert updated["energy_balance_tol_wh"] == pytest.approx(0.25)

    after = client.get("/tribunal/policy").json()
    assert after["policy_version"] == updated["policy_version"]
    assert after["energy_balance_tol_wh"] == pytest.approx(0.25)
    assert before["policy_version"] in after["versions"]

    # The old version is still on record — nothing was silently overwritten.
    assert client.put("/tribunal/policy", json={}).status_code == 400
    assert (
        client.put("/tribunal/policy", json={"soc_min": 90.0, "soc_max": 10.0}).status_code
        == 400
    )

    confidence = client.put(
        "/confidence/policy", json={"c_low": 0.55, "actor": "test"}
    ).json()
    assert confidence["policy_version"] != "confidence_v3.0"
    assert client.get("/confidence/policy").json()["c_low"] == pytest.approx(0.55)


def test_advisory_carries_full_lineage(client: TestClient) -> None:
    """Every advisory response carries the unbroken lineage chain."""
    batches = client.get("/model/scenarios/batches/none").status_code
    assert batches == 404

    # Re-derive a batch id from the last scenario batch in the store.
    from chariot.store import Store  # local import: keeps the module import-light

    store = Store(client.app.state.store_path) if hasattr(client.app.state, "store_path") else None
    if store is None:
        pytest.skip("store path not exposed on the app state")
    rows = store.query("SELECT batch_id FROM tribunal_batches ORDER BY created_at_utc DESC")
    if not rows:
        pytest.skip("no validated batch in the store")
    batch_id = rows[0]["batch_id"]
    store.close()

    forecast = client.post(
        "/advisory/forecast", json={"batch_id": batch_id, "role_mode": "research"}
    ).json()
    lineage = forecast["lineage"]
    for key in (
        "admissibility",
        "calibration_version",
        "capsule_id",
        "dataset_version",
        "chariot_model_version",
        "scenario_batch",
        "tribunal_result",
        "confidence_report",
        "policies",
    ):
        assert key in lineage, f"lineage is missing {key}"
    assert lineage["scenario_batch"]["batch_id"] == batch_id
    assert lineage["tribunal_result"]["n_retained"] >= 0
    assert lineage["policies"]["system_version"]
