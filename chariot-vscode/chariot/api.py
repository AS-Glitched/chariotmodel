"""CHARIOT v3 HTTP surface (PDF §4.15).

The API is **advisory only**: there is deliberately no control, dispatch or
setpoint endpoint anywhere in this module.  Every response is a certified
read-out of the lineage store or of an in-memory run artefact.

Endpoint map (PDF §4.15)::

    /sites                              /model/chariot/train
    /sites/{site_id}/context            /model/chariot/infer
    /admissibility/evaluate             /model/chariot/scenarios
    /admissibility/decisions            /model/chariot/versions
    /admissibility/rule-sets            /model/chariot/promote
    /flow/ingest                        /model/residual/train
    /slices/vertical                    /model/residual/propose
    /slices/horizontal                  /model/residual/versions
    /replay/{replay_id}                 /model/scenarios/generate
    /calibration/versions               /model/scenarios/batches/{batch_id}
    /calibration/resolve                /tribunal/validate-batch
    /capsules/seal                      /tribunal/batches/{batch_id}/retained
    /training-extract/build             /tribunal/batches/{batch_id}/rejections
    /training-extract/datasets/{id}     /tribunal/policy
    /confidence/evaluate                /feedback/cortex/run
    /confidence/reports/{report_id}     /feedback/jobs
    /confidence/policy                  /residual/governor/status
    /advisory/forecast                  /residual/governor/evaluate
    /advisory/events                    /residual/governor/quarantine
    /advisory/confidence                /residual/governor/set-weight
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from chariot import admissibility as adm
from chariot import advisory as adv
from chariot import calibration as cal
from chariot import capsule as cap
from chariot import confidence as conf
from chariot import feedback as fb
from chariot import genesis as gen
from chariot import governor as gov
from chariot import metrics as met
from chariot import model as mdl
from chariot import residual as res
from chariot import river_flow as flow
from chariot import site_master as sm
from chariot import training_extract as tx
from chariot import tribunal as trib
from chariot.config import (
    ADMISSIBILITY_POLICY,
    CONFIDENCE_POLICY,
    GOVERNOR_POLICY,
    HORIZONS,
    RESIDUAL_POLICY,
    TRIBUNAL_POLICY,
    SiteContext,
    next_policy_version,
    policy_snapshot,
)
from chariot.store import Store, utc_now

API_TITLE = "CHARIOT v3"
API_VERSION = "3.0.0"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    """In-memory state of the running API (the durable state is the store)."""

    store: Store
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, mdl.ChariotModel] = field(default_factory=dict)
    batches: dict[str, gen.ScenarioBatch] = field(default_factory=dict)
    reports: dict[str, conf.ConfidenceReport] = field(default_factory=dict)
    advisories: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, SiteContext] = field(default_factory=dict)
    governor: gov.ResidualStabilityGovernor | None = None
    retained: dict[str, Any] = field(default_factory=dict)
    #: Effective policies.  A PUT on /tribunal/policy or /confidence/policy
    #: mints a new version and replaces the effective one; the previous
    #: version stays on record in ``policy_versions``.
    tribunal_policy: Any = TRIBUNAL_POLICY
    confidence_policy: Any = CONFIDENCE_POLICY

    def frame_for(self, tenant_id: str, site_id: str) -> pd.DataFrame:
        key = f"{tenant_id}/{site_id}"
        if key not in self.frames:
            raise HTTPException(status_code=404, detail=f"no ingested telemetry for {key}")
        return self.frames[key]

    def context_for(self, tenant_id: str, site_id: str) -> SiteContext:
        key = f"{tenant_id}/{site_id}"
        if key not in self.context:
            self.context[key] = sm.resolve_context(self.store, tenant_id, site_id)
        return self.context[key]

    def ensure_governor(self) -> gov.ResidualStabilityGovernor:
        if self.governor is None:
            self.governor = gov.ResidualStabilityGovernor(
                store=self.store, w_max=RESIDUAL_POLICY.w_max
            )
        return self.governor


def _frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="no telemetry records supplied")
    frame = pd.DataFrame(records)
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    for column in ("solar_w", "clear_sky_w", "load_w", "soc_percent"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SiteRequest(BaseModel):
    tenant_id: str
    site_id: str
    name: str = ""
    status: str = "active"
    rated_pv_w: float | None = None
    usable_battery_capacity_wh: float | None = None
    max_charge_w: float | None = None
    max_discharge_w: float | None = None
    timezone: str = "UTC"
    latitude: float | None = None
    longitude: float | None = None
    fleet_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    tenant_id: str
    site_id: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    feature_set: str = "solar_only_v1"


class IngestRequest(BaseModel):
    tenant_id: str
    site_id: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    horizontal_parameters: list[str] = Field(default_factory=lambda: ["solar_w", "clear_sky_w"])
    issue_replay: bool = True


class CalibrationRequest(BaseModel):
    scope_type: str = "site"
    scope_id: str = "default"
    capacity_factor: float = 1.0
    charge_loss_factor: float = 0.0
    discharge_loss_factor: float = 0.0
    pv_correction_factor: float = 1.0
    method: str = "engineering_declaration"
    notes: str = ""
    tenant_id: str | None = None
    site_id: str | None = None
    created_by: str = "api"


class SealRequest(BaseModel):
    tenant_id: str
    site_id: str
    calibration_version: str | None = None
    slice_ids: list[str] = Field(default_factory=list)
    slice_hashes: list[str] = Field(default_factory=list)
    admissibility_versions: list[str] = Field(default_factory=list)
    require_clear_sky: bool = True
    created_by: str = "api"


class ExtractRequest(BaseModel):
    tenant_id: str
    site_id: str
    capsule_ids: list[str] = Field(default_factory=list)
    calibration_version: str | None = None
    calibration_factors: dict[str, Any] = Field(default_factory=dict)
    admissibility_versions: list[str] = Field(default_factory=list)
    slice_ids: list[str] = Field(default_factory=list)
    train_frac: float = 0.6
    validate_frac: float = 0.2
    test_frac: float = 0.2
    created_by: str = "api"


class TrainRequest(BaseModel):
    tenant_id: str
    site_id: str
    dataset_version: str | None = None
    fleet_id: str = "fleet"
    n_scenarios: int = 500
    seed: int = 7
    horizon: str = "operational"
    use_residual: bool = True
    created_by: str = "api"


class ScenarioRequest(BaseModel):
    tenant_id: str
    site_id: str
    model_version: str | None = None
    n_scenarios: int = 500
    seed: int = 7
    horizon: str = "operational"
    records: list[dict[str, Any]] = Field(default_factory=list)
    origin_utc: str | None = None


class ResidualTrainRequest(BaseModel):
    tenant_id: str
    site_id: str
    dataset_version: str | None = None
    alpha: float = 10.0
    target: str = "solar_w"
    w_max: float = RESIDUAL_POLICY.w_max
    created_by: str = "api"


class TribunalRequest(BaseModel):
    batch_id: str


class CortexRequest(BaseModel):
    batch_id: str
    survival_rate: float | None = None
    survival_delta: float = 0.0


class GovernorRequest(BaseModel):
    batch_id: str | None = None
    w: float | None = None
    actor: str = "api"
    quarantine: bool = False
    reason: str = ""


class ConfidenceRequest(BaseModel):
    batch_id: str
    residual_context: dict[str, Any] = Field(default_factory=dict)


class AdvisoryRequest(BaseModel):
    batch_id: str
    role_mode: str = "research"
    horizon: str | None = None
    horizon_steps: int | None = None


class PolicyRequest(BaseModel):
    """A *new* version of a policy.

    Policies are never edited in place: a PUT mints the next version and the
    previous one stays on record.  Every field is optional — an omitted field
    keeps the currently effective value.
    """

    actor: str = "api"
    reason: str = ""
    energy_balance_tol_wh: float | None = None
    soc_min: float | None = None
    soc_max: float | None = None
    solar_scale_kappa: float | None = None
    power_limit_tolerance: float | None = None
    solar_ramp_w_per_step: float | None = None
    load_ramp_w_per_step: float | None = None
    min_survival_for_certified: float | None = None
    s_min: float | None = None
    c_low: float | None = None
    u_high: float | None = None
    sharpness: str | None = None
    uncertainty_reference: float | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(store_path: str | Path | None = None) -> FastAPI:
    """Build the CHARIOT v3 FastAPI application."""
    path = store_path or os.environ.get("CHARIOT_STORE", "var/chariot.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    runtime = Runtime(store=store)

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "CHARIOT v3 — physics-governed hierarchical energy intelligence. "
            "Advisory only: no control, dispatch or setpoint endpoint exists."
        ),
    )
    app.state.runtime = runtime
    app.state.store_path = str(path)

    # Seed the built-in policies into the version history so the audit trail
    # is complete from the first version onwards.
    for policy_type, policy in (("tribunal", TRIBUNAL_POLICY), ("confidence", CONFIDENCE_POLICY)):
        if not store.policy_versions(policy_type):
            store.record_policy_version(
                policy_type, policy.policy_version, policy.to_dict(), actor="system"
            )

    # ---------------------------------------------------------------- sites
    @app.get("/sites")
    def list_sites(tenant_id: str | None = Query(default=None)) -> dict[str, Any]:
        sites = store.get_sites(tenant_id)
        return {"count": len(sites), "sites": sites}

    @app.post("/sites", status_code=201)
    def put_site(request: SiteRequest) -> dict[str, Any]:
        payload = sm.upsert_site(
            store,
            request.tenant_id,
            request.site_id,
            name=request.name,
            status=request.status,
            rated_pv_w=request.rated_pv_w,
            usable_battery_capacity_wh=request.usable_battery_capacity_wh,
            max_charge_w=request.max_charge_w,
            max_discharge_w=request.max_discharge_w,
            timezone=request.timezone,
            latitude=request.latitude,
            longitude=request.longitude,
            fleet_id=request.fleet_id,
            metadata=request.metadata,
            updated_by="api",
        )
        runtime.context.pop(f"{request.tenant_id}/{request.site_id}", None)
        return payload

    @app.get("/sites/{site_id}/context")
    def site_context(
        site_id: str, tenant_id: str = Query(...)
    ) -> dict[str, Any]:
        ctx = runtime.context_for(tenant_id, site_id)
        return sm.context_payload(ctx)

    # -------------------------------------------------------- admissibility
    @app.post("/admissibility/evaluate")
    def admissibility_evaluate(request: EvaluateRequest) -> dict[str, Any]:
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        frame = (
            _frame_from_records(request.records)
            if request.records
            else runtime.frame_for(request.tenant_id, request.site_id)
        )
        decisions = adm.evaluate_frame(
            frame, ctx=ctx, policy=ADMISSIBILITY_POLICY, feature_set=request.feature_set
        )
        rows = decisions.to_dict("records")
        stamps = [str(v) for v in frame["timestamp_utc"].tolist()]
        for stamp, row in zip(stamps, rows):
            store.insert(
                "admissibility_decisions",
                {
                    "decision_id": f"adm_{stamp}_{request.site_id}",
                    "tenant_id": request.tenant_id,
                    "site_id": request.site_id,
                    "timestamp_utc": stamp,
                    "decision": row.get("decision"),
                    "hard_failures": list(row.get("hard_failures", [])),
                    "soft_warnings": list(row.get("soft_warnings", [])),
                    "rule_set_version": row.get(
                        "rule_set_version", ADMISSIBILITY_POLICY.rule_set_version
                    ),
                    "input_fingerprint": row.get("input_fingerprint", ""),
                    "created_at_utc": utc_now(),
                },
            )
        return {
            "rule_set_version": ADMISSIBILITY_POLICY.rule_set_version,
            "summary": adm.summarize(rows),
            "decisions": rows,
        }

    @app.get("/admissibility/decisions")
    def admissibility_decisions(
        site_id: str | None = Query(default=None),
        limit: int = Query(default=200, le=2000),
    ) -> dict[str, Any]:
        sql = "SELECT * FROM admissibility_decisions"
        params: list[Any] = []
        if site_id:
            sql += " WHERE site_id = ?"
            params.append(site_id)
        sql += " ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)
        rows = store.query(sql, tuple(params))
        return {"count": len(rows), "decisions": rows}

    @app.get("/admissibility/rule-sets")
    def admissibility_rule_sets() -> dict[str, Any]:
        return {
            "admissibility": ADMISSIBILITY_POLICY.to_dict(),
            "tribunal": TRIBUNAL_POLICY.to_dict(),
            "confidence": CONFIDENCE_POLICY.to_dict(),
            "residual": RESIDUAL_POLICY.to_dict(),
            "governor": GOVERNOR_POLICY.to_dict(),
        }

    # ------------------------------------------------------------ river flow
    @app.post("/flow/ingest")
    def flow_ingest(request: IngestRequest) -> dict[str, Any]:
        frame = (
            _frame_from_records(request.records)
            if request.records
            else runtime.frame_for(request.tenant_id, request.site_id)
        )
        runtime.frames[f"{request.tenant_id}/{request.site_id}"] = frame
        vertical = flow.build_vertical_slice(
            store, frame, tenant_id=request.tenant_id, site_id=request.site_id
        )
        horizontal = [
            flow.build_horizontal_slice(
                store,
                frame,
                parameter,
                tenant_id=request.tenant_id,
                site_id=request.site_id,
            )
            for parameter in request.horizontal_parameters
            if parameter in frame.columns
        ]
        slice_ids = [vertical["slice_id"]] + [h["slice_id"] for h in horizontal]
        slice_hashes = [vertical["content_hash"]] + [h["content_hash"] for h in horizontal]
        replay = None
        if request.issue_replay and slice_ids:
            replay = flow.issue_replay_certificate(
                store,
                slice_ids,
                tenant_id=request.tenant_id,
                site_id=request.site_id,
            )
        return {
            "rows": int(len(frame)),
            "vertical_slice": vertical,
            "horizontal_slices": horizontal,
            "slice_ids": slice_ids,
            "slice_hashes": slice_hashes,
            "replay": replay,
        }

    @app.get("/slices/vertical")
    def slices_vertical(
        tenant_id: str = Query(...), site_id: str = Query(...)
    ) -> dict[str, Any]:
        rows = store.query(
            "SELECT * FROM river_slices WHERE tenant_id = ? AND site_id = ? AND slice_type = ?",
            (tenant_id, site_id, "vertical"),
        )
        return {"count": len(rows), "slices": rows}

    @app.get("/slices/horizontal")
    def slices_horizontal(
        tenant_id: str = Query(...), site_id: str = Query(...)
    ) -> dict[str, Any]:
        rows = store.query(
            "SELECT * FROM river_slices WHERE tenant_id = ? AND site_id = ? AND slice_type = ?",
            (tenant_id, site_id, "horizontal"),
        )
        return {"count": len(rows), "slices": rows}

    @app.get("/replay/{replay_id}")
    def replay(replay_id: str) -> dict[str, Any]:
        return flow.verify_replay(store, replay_id)

    # ---------------------------------------------------------- calibration
    @app.get("/calibration/versions")
    def calibration_versions(
        site_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        sql = "SELECT * FROM calibration_versions"
        params: list[Any] = []
        if site_id:
            sql += " WHERE scope_type = 'site' AND scope_id = ?"
            params.append(site_id)
        sql += " ORDER BY created_at_utc DESC"
        return {"count": len(store.query(sql, tuple(params))), "versions": store.query(sql, tuple(params))}

    @app.post("/calibration/versions", status_code=201)
    def calibration_create(request: CalibrationRequest) -> dict[str, Any]:
        version = cal.create_version(
            store,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            capacity_factor=request.capacity_factor,
            charge_loss_factor=request.charge_loss_factor,
            discharge_loss_factor=request.discharge_loss_factor,
            pv_correction_factor=request.pv_correction_factor,
            method=request.method,
            notes=request.notes,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            created_by=request.created_by,
        )
        return {"calibration_version": version, "status": "draft"}

    @app.post("/calibration/versions/{calibration_version}/approve")
    def calibration_approve(calibration_version: str) -> dict[str, Any]:
        cal.approve_version(store, calibration_version, approved_by="api")
        return {"calibration_version": calibration_version, "status": "approved"}

    @app.post("/calibration/versions/{calibration_version}/retire")
    def calibration_retire(calibration_version: str) -> dict[str, Any]:
        cal.retire_version(store, calibration_version, actor="api")
        return {"calibration_version": calibration_version, "status": "retired"}

    @app.get("/calibration/resolve")
    def calibration_resolve(
        tenant_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
        fleet_id: str | None = Query(default=None),
        equipment_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        factors = cal.resolve(
            store,
            tenant_id=tenant_id,
            site_id=site_id,
            fleet_id=fleet_id,
            equipment_id=equipment_id,
        )
        return factors.to_dict()

    # -------------------------------------------------------------- capsules
    @app.post("/capsules/seal", status_code=201)
    def capsules_seal(request: SealRequest) -> dict[str, Any]:
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        frame = runtime.frame_for(request.tenant_id, request.site_id)
        sealed = cap.seal_capsule(
            store,
            frame,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            calibration_version=request.calibration_version,
            admissibility_versions=request.admissibility_versions,
            slice_ids=request.slice_ids,
            slice_hashes=request.slice_hashes,
            ctx=ctx,
            created_by=request.created_by,
            require_clear_sky=request.require_clear_sky,
        )
        return sealed

    @app.get("/capsules/{capsule_id}")
    def capsule_verify(capsule_id: str) -> dict[str, Any]:
        return cap.verify_capsule(store, capsule_id)

    # ------------------------------------------------------ training extract
    @app.post("/training-extract/build", status_code=201)
    def training_extract_build(request: ExtractRequest) -> dict[str, Any]:
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        frame = runtime.frame_for(request.tenant_id, request.site_id)
        factors = {
            k: float(v)
            for k, v in (request.calibration_factors or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        manifest = tx.build_training_extract(
            store,
            frame,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            capsule_ids=request.capsule_ids,
            calibration_version=request.calibration_version,
            calibration_factors=factors,
            admissibility_versions=request.admissibility_versions,
            slice_ids=request.slice_ids,
            ctx=ctx,
            train_frac=request.train_frac,
            validate_frac=request.validate_frac,
            test_frac=request.test_frac,
            created_by=request.created_by,
        )
        return manifest

    @app.get("/training-extract/datasets/{dataset_version}")
    def training_extract_dataset(dataset_version: str) -> dict[str, Any]:
        manifest = tx.get_manifest(store, dataset_version)
        frame = tx.load_dataset(store, dataset_version)
        return {
            "manifest": manifest,
            "rows": int(len(frame)),
            "split_counts": frame["split"].value_counts().to_dict(),
        }

    # ----------------------------------------------------------- chariot model
    @app.post("/model/chariot/train", status_code=201)
    def model_train(request: TrainRequest) -> dict[str, Any]:
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        if request.dataset_version:
            dataset = tx.load_dataset(store, request.dataset_version)
        else:
            frame = runtime.frame_for(request.tenant_id, request.site_id)
            manifest = tx.build_training_extract(
                store,
                frame,
                tenant_id=request.tenant_id,
                site_id=request.site_id,
                capsule_ids=[],
                calibration_version=None,
                calibration_factors={},
                admissibility_versions=[],
                slice_ids=[],
                ctx=ctx,
                created_by=request.created_by,
            )
            dataset = tx.load_dataset(store, manifest["dataset_version"])
        train = dataset.loc[dataset["split"] == "train"].reset_index(drop=True)
        validate = dataset.loc[dataset["split"] == "validate"].reset_index(drop=True)
        model = mdl.fit(
            train,
            ctx,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            fleet_id=request.fleet_id,
            dataset_version=request.dataset_version or "",
            validation=validate if len(validate) else None,
        )
        runtime.models[model.model_version] = model
        store.upsert(
            "model_versions",
            {
                "model_version": model.model_version,
                "model_name": model.model_name,
                "dataset_version": request.dataset_version or "",
                "dataset_content_hash": "",
                "params_json": json.dumps(model.to_dict(), sort_keys=True, default=str),
                "diagnostics_json": json.dumps(model.diagnostics, sort_keys=True, default=str),
                "status": "candidate",
                "created_at_utc": utc_now(),
            },
        )
        return {
            "model_version": model.model_version,
            "model_name": model.model_name,
            "solar_channel": model.sites[request.site_id].solar.to_dict(),
            "diagnostics": model.diagnostics,
        }

    @app.post("/model/chariot/infer")
    def model_infer(request: ScenarioRequest) -> dict[str, Any]:
        model = _model_for(runtime, request.model_version, request.site_id)
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        frame = (
            _frame_from_records(request.records)
            if request.records
            else runtime.frame_for(request.tenant_id, request.site_id)
        )
        prediction = mdl.one_step_prediction(model, request.site_id, frame)
        truth = mdl.solar_series(frame)
        mask = np.isfinite(truth) & np.isfinite(prediction)
        return {
            "model_version": model.model_version,
            "n_rows": int(mask.sum()),
            "one_step_mae_w": float(np.mean(np.abs(prediction[mask] - truth[mask])))
            if mask.any()
            else None,
            "one_step_rmse_w": float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2)))
            if mask.any()
            else None,
            "timestamps": [str(v) for v in frame["timestamp_utc"][mask].tolist()],
            "prediction_w": [round(float(v), 3) for v in prediction[mask]],
        }

    @app.post("/model/chariot/scenarios")
    def model_scenarios(request: ScenarioRequest) -> dict[str, Any]:
        return _generate_batch(runtime, request)

    @app.get("/model/chariot/versions")
    def model_versions() -> dict[str, Any]:
        rows = store.query("SELECT * FROM model_versions ORDER BY created_at_utc DESC")
        return {"count": len(rows), "versions": rows}

    @app.post("/model/chariot/promote")
    def model_promote(model_version: str = Body(..., embed=True)) -> dict[str, Any]:
        store.execute(
            "UPDATE model_versions SET status = 'active' WHERE model_version = ?",
            (model_version,),
        )
        store.execute(
            "UPDATE model_versions SET status = 'superseded' "
            "WHERE model_version != ? AND status = 'active'",
            (model_version,),
        )
        return {"model_version": model_version, "status": "active"}

    # -------------------------------------------------------------- residual
    @app.post("/model/residual/train", status_code=201)
    def residual_train(request: ResidualTrainRequest) -> dict[str, Any]:
        if request.dataset_version:
            dataset = tx.load_dataset(store, request.dataset_version)
        else:
            raise HTTPException(status_code=400, detail="dataset_version is required")
        train = dataset.loc[dataset["split"] == "train"].reset_index(drop=True)
        validate = dataset.loc[dataset["split"] == "validate"].reset_index(drop=True)
        ctx = runtime.context_for(request.tenant_id, request.site_id)
        model = _model_for(runtime, None, request.site_id)
        mu = mdl.one_step_prediction(model, request.site_id, train)
        clear = mdl.clear_sky_series(train)
        y = mdl.solar_series(train)
        expected = np.divide(
            mu, np.maximum(clear, 1.0), out=np.full_like(mu, np.nan), where=clear > 1.0
        )
        X, names = res.build_features(
            train,
            mu_solar=mu,
            mu_load=np.zeros(len(train)),
            clear_sky=clear,
            expected_ratio=expected,
            last_solar=float(y[0]) if len(y) else 0.0,
        )
        artifact = res.fit_ridge(
            X,
            y,
            alpha=request.alpha,
            feature_names=names,
            target=request.target,
            dataset_version=request.dataset_version,
            w_max=request.w_max,
        )
        res.register(store, artifact, status="draft")
        return {"residual_version": artifact.residual_version, "diagnostics": artifact.diagnostics}

    @app.post("/model/residual/propose")
    def residual_propose(
        artifact: dict[str, Any] = Body(...), X: list[list[float]] = Body(...)
    ) -> dict[str, Any]:
        from chariot.residual import ResidualArtifact

        rebuilt = ResidualArtifact(
            residual_version=artifact.get("residual_version", "ad_hoc"),
            family=artifact.get("family", "ridge"),
            target=artifact.get("target", "solar_w"),
            feature_names=artifact.get("feature_names", []),
            coefficients=artifact.get("coefficients", []),
            intercept=float(artifact.get("intercept", 0.0)),
            w_max=float(artifact.get("w_max", RESIDUAL_POLICY.w_max)),
            diagnostics=artifact.get("diagnostics", {}),
        )
        paths = res.propose_paths(rebuilt, np.asarray(X, dtype=float))
        return {"residual_version": rebuilt.residual_version, "paths": paths.tolist()}

    @app.get("/model/residual/versions")
    def residual_versions() -> dict[str, Any]:
        rows = store.query("SELECT * FROM residual_versions ORDER BY created_at_utc DESC")
        return {"count": len(rows), "versions": rows}

    # -------------------------------------------------------------- scenarios
    @app.post("/model/scenarios/generate", status_code=201)
    def scenarios_generate(request: ScenarioRequest) -> dict[str, Any]:
        return _generate_batch(runtime, request)

    @app.get("/model/scenarios/batches/{batch_id}")
    def scenarios_batch(batch_id: str) -> dict[str, Any]:
        batch = runtime.batches.get(batch_id)
        if batch is None:
            row = store.query_one(
                "SELECT * FROM scenario_batches WHERE batch_id = ?", (batch_id,)
            )
            if row is None:
                raise HTTPException(status_code=404, detail=f"unknown batch {batch_id}")
            return {"batch_id": batch_id, "persisted": row}
        return {
            "batch_id": batch.batch_id,
            "origin_utc": batch.origin_utc,
            "n_scenarios": batch.n_scenarios,
            "horizon_steps": batch.horizon_steps,
            "content_hash": batch.content_hash,
            "timestamps": [str(v) for v in batch.timestamps],
        }

    # --------------------------------------------------------------- tribunal
    @app.post("/tribunal/validate-batch")
    def tribunal_validate(request: TribunalRequest) -> dict[str, Any]:
        batch = runtime.batches.get(request.batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"unknown batch {request.batch_id}")
        ctx = runtime.context_for(batch.tenant_id, batch.site_id)
        policy = runtime.tribunal_policy
        result = trib.validate_batch(
            batch,
            ctx=ctx,
            calibration=batch.lineage.get("calibration"),
            policy=policy,
            store=store,
        )
        runtime.retained[batch.batch_id] = result
        return {
            "batch_id": batch.batch_id,
            "n_proposed": result.n_proposed,
            "n_retained": result.n_retained,
            "n_rejected": result.n_rejected,
            "survival_rate": result.survival_rate,
            "constraint_counts": result.constraint_counts,
            "policy_version": policy.policy_version,
        }

    @app.get("/tribunal/batches/{batch_id}/retained")
    def tribunal_retained(batch_id: str) -> dict[str, Any]:
        result = runtime.retained.get(batch_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no tribunal result for {batch_id}")
        return {
            "batch_id": batch_id,
            "n_retained": result.n_retained,
            "retained": [batch_id + "_" + str(i) for i in result.retained_idx],
        }

    @app.get("/tribunal/rejections")
    def tribunal_rejections(
        batch_id: str | None = Query(default=None),
        limit: int = Query(default=100, le=1000),
    ) -> dict[str, Any]:
        """The Rejection Echo — append-only, never pruned."""
        if batch_id:
            rows = store.rejections_for_batch(batch_id)[:limit]
        else:
            rows = store.all_rejections(limit=limit)
        return {"batch_id": batch_id, "count": len(rows), "rejections": rows}

    @app.get("/tribunal/rejections/{rejection_id}")
    def tribunal_rejection(rejection_id: str) -> dict[str, Any]:
        row = store.get_rejection(rejection_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown rejection {rejection_id}")
        return row

    @app.get("/tribunal/batches/{batch_id}/rejections")
    def tribunal_batch_rejections(
        batch_id: str, limit: int = Query(default=100, le=1000)
    ) -> dict[str, Any]:
        rows = store.rejections_for_batch(batch_id)[:limit]
        return {"batch_id": batch_id, "count": len(rows), "rejections": rows}

    @app.get("/tribunal/policy")
    def tribunal_policy() -> dict[str, Any]:
        policy = runtime.tribunal_policy
        return {
            **policy.to_dict(),
            "effective": True,
            "versions": [row["policy_version"] for row in store.policy_versions("tribunal")],
        }

    @app.put("/tribunal/policy")
    def tribunal_policy_put(request: PolicyRequest) -> dict[str, Any]:
        """Mint a new versioned Tribunal policy.

        The current policy is never mutated in place — a change creates the
        next version, records it, and makes it effective.
        """
        current = runtime.tribunal_policy
        updates = {
            key: value
            for key, value in request.model_dump(exclude_none=True).items()
            if key in {
                "energy_balance_tol_wh",
                "soc_min",
                "soc_max",
                "solar_scale_kappa",
                "power_limit_tolerance",
                "solar_ramp_w_per_step",
                "load_ramp_w_per_step",
                "min_survival_for_certified",
            }
        }
        if not updates:
            raise HTTPException(status_code=400, detail="no policy fields supplied")
        if updates.get("soc_min") is not None and updates.get("soc_max") is not None:
            if float(updates["soc_min"]) >= float(updates["soc_max"]):
                raise HTTPException(status_code=400, detail="soc_min must be < soc_max")
        version = next_policy_version(current.policy_version, "tribunal")
        new_policy = replace(current, policy_version=version, **updates)
        store.record_policy_version(
            "tribunal",
            version,
            {**new_policy.to_dict(), "reason": request.reason},
            actor=request.actor,
        )
        runtime.tribunal_policy = new_policy
        return {
            **new_policy.to_dict(),
            "previous_version": current.policy_version,
            "superseded": True,
        }

    # ---------------------------------------------------------------- feedback
    @app.post("/feedback/cortex/run")
    def feedback_run(request: CortexRequest) -> dict[str, Any]:
        result = runtime.retained.get(request.batch_id)
        survival = (
            request.survival_rate
            if request.survival_rate is not None
            else (result.survival_rate if result is not None else 0.0)
        )
        job = fb.run_cortex(
            store,
            batch_id=request.batch_id,
            governor=runtime.ensure_governor(),
            survival_rate=survival,
            survival_delta=request.survival_delta,
        )
        return job

    @app.get("/feedback/jobs")
    def feedback_jobs(limit: int = Query(default=50, le=500)) -> dict[str, Any]:
        rows = store.query(
            "SELECT * FROM feedback_jobs ORDER BY created_at_utc DESC LIMIT ?", (limit,)
        )
        return {"count": len(rows), "jobs": rows}

    # --------------------------------------------------------------- governor
    @app.get("/residual/governor/status")
    def governor_status() -> dict[str, Any]:
        return runtime.ensure_governor().status()

    @app.post("/residual/governor/evaluate")
    def governor_evaluate(request: GovernorRequest) -> dict[str, Any]:
        batch = runtime.batches.get(request.batch_id or "")
        if batch is None:
            raise HTTPException(status_code=404, detail="batch_id is required")
        ctx = runtime.context_for(batch.tenant_id, batch.site_id)
        return gov.governor_signals(
            runtime.ensure_governor(),
            batch=batch,
            ctx=ctx,
            store=store,
        )

    @app.post("/residual/governor/quarantine")
    def governor_quarantine(request: GovernorRequest) -> dict[str, Any]:
        runtime.ensure_governor().quarantine(
            batch_id=request.batch_id or "", reason=request.reason, actor=request.actor
        )
        return runtime.ensure_governor().status()

    @app.post("/residual/governor/set-weight")
    def governor_set_weight(request: GovernorRequest) -> dict[str, Any]:
        if request.w is None:
            raise HTTPException(status_code=400, detail="w is required")
        runtime.ensure_governor().set_weight(float(request.w), actor=request.actor)
        return runtime.ensure_governor().status()

    # -------------------------------------------------------------- confidence
    @app.post("/confidence/evaluate")
    def confidence_evaluate(request: ConfidenceRequest) -> dict[str, Any]:
        batch = runtime.batches.get(request.batch_id)
        result = runtime.retained.get(request.batch_id)
        if batch is None or result is None:
            raise HTTPException(status_code=404, detail="batch has not been validated")
        policy = runtime.confidence_policy
        report = conf.evaluate(
            batch,
            list(result.retained_idx),
            survival_rate=result.survival_rate,
            policy=policy,
            residual_context=request.residual_context or None,
            store=store,
        )
        runtime.reports[report.report_id] = report
        return report.to_dict()

    @app.get("/confidence/reports/{report_id}")
    def confidence_report(report_id: str) -> dict[str, Any]:
        report = runtime.reports.get(report_id)
        if report is not None:
            return report.to_dict()
        row = store.query_one("SELECT * FROM confidence_reports WHERE report_id = ?", (report_id,))
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown report {report_id}")
        return row

    @app.get("/confidence/policy")
    def confidence_policy() -> dict[str, Any]:
        policy = runtime.confidence_policy
        return {
            **policy.to_dict(),
            "effective": True,
            "versions": [row["policy_version"] for row in store.policy_versions("confidence")],
        }

    @app.put("/confidence/policy")
    def confidence_policy_put(request: PolicyRequest) -> dict[str, Any]:
        """Mint a new versioned Confidence Engine policy."""
        current = runtime.confidence_policy
        updates = {
            key: value
            for key, value in request.model_dump(exclude_none=True).items()
            if key in {"s_min", "c_low", "u_high", "sharpness", "uncertainty_reference"}
        }
        if not updates:
            raise HTTPException(status_code=400, detail="no policy fields supplied")
        if updates.get("s_min") is not None and not 0.0 <= float(updates["s_min"]) <= 1.0:
            raise HTTPException(status_code=400, detail="s_min must be in [0, 1]")
        version = next_policy_version(current.policy_version, "confidence")
        new_policy = replace(current, policy_version=version, **updates)
        store.record_policy_version(
            "confidence",
            version,
            {**new_policy.to_dict(), "reason": request.reason},
            actor=request.actor,
        )
        runtime.confidence_policy = new_policy
        return {
            **new_policy.to_dict(),
            "previous_version": current.policy_version,
            "superseded": True,
        }

    # ---------------------------------------------------------------- advisory
    @app.post("/advisory/forecast")
    def advisory_forecast(request: AdvisoryRequest) -> dict[str, Any]:
        batch = runtime.batches.get(request.batch_id)
        result = runtime.retained.get(request.batch_id)
        report = None
        for candidate in runtime.reports.values():
            if candidate.batch_id == request.batch_id:
                report = candidate
                break
        if batch is None or result is None or report is None:
            raise HTTPException(
                status_code=404, detail="batch has not been validated/confidenced"
            )
        ctx = runtime.context_for(batch.tenant_id, batch.site_id)
        payload = adv.build_forecast(
            batch,
            list(result.retained_idx),
            report,
            role_mode=request.role_mode,
            horizon=request.horizon,
            ctx=ctx,
            horizon_steps=request.horizon_steps,
            lineage=_advisory_lineage(runtime, batch, result, report),
        )
        runtime.advisories[request.batch_id] = payload
        return payload

    @app.get("/advisory/events")
    def advisory_events(batch_id: str = Query(...)) -> dict[str, Any]:
        batch = runtime.batches.get(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"unknown batch {batch_id}")
        ctx = runtime.context_for(batch.tenant_id, batch.site_id)
        result = runtime.retained.get(batch_id)
        retained_idx = list(result.retained_idx) if result is not None else []
        return adv.build_events(batch, retained_idx, ctx=ctx)

    @app.get("/advisory/confidence")
    def advisory_confidence(batch_id: str = Query(...)) -> dict[str, Any]:
        for report in runtime.reports.values():
            if report.batch_id == batch_id:
                return report.to_dict()
        raise HTTPException(status_code=404, detail=f"no confidence report for {batch_id}")

    # ----------------------------------------------------------------- meta
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "store": str(path),
            "policies": policy_snapshot(),
            "horizons": dict(HORIZONS),
            "advisory_only": True,
        }

    return app


def _model_for(
    runtime: Runtime, model_version: str | None, site_id: str
) -> mdl.ChariotModel:
    if model_version:
        model = runtime.models.get(model_version)
        if model is None:
            raise HTTPException(status_code=404, detail=f"unknown model {model_version}")
        return model
    for model in reversed(list(runtime.models.values())):
        if site_id in model.sites:
            return model
    raise HTTPException(status_code=404, detail="no trained model in memory")


def _advisory_lineage(
    runtime: Runtime,
    batch: gen.ScenarioBatch,
    result: Any,
    report: conf.ConfidenceReport,
) -> dict[str, Any]:
    """Full lineage chain carried on every advisory response (PDF §4.14).

    telemetry → admissibility → river slice → calibration → capsule →
    dataset → model → residual → scenario batch → tribunal result →
    confidence report → advisory
    """
    store = runtime.store
    batch_lineage = batch.lineage or {}
    calibration = batch_lineage.get("calibration") or {}

    slice_ids: list[str] = []
    capsule_ids: list[str] = []
    try:
        for row in store.query(
            "SELECT slice_id FROM river_slices WHERE tenant_id = ? AND site_id = ?"
            " ORDER BY slice_id",
            (batch.tenant_id, batch.site_id),
        ):
            slice_ids.append(row["slice_id"])
        for row in store.query(
            "SELECT capsule_id FROM capsules WHERE tenant_id = ? AND site_id = ?"
            " ORDER BY capsule_id",
            (batch.tenant_id, batch.site_id),
        ):
            capsule_ids.append(row["capsule_id"])
    except Exception:  # pragma: no cover - lineage is best-effort, never fatal
        slice_ids, capsule_ids = [], []

    return {
        "tenant_id": batch.tenant_id,
        "site_id": batch.site_id,
        "admissibility": {
            "rule_set_version": ADMISSIBILITY_POLICY.rule_set_version,
        },
        "river_slice": {"slice_ids": slice_ids},
        "calibration_version": calibration.get("calibration_version"),
        "calibration_factors": {
            key: calibration.get(key)
            for key in (
                "capacity_factor",
                "charge_loss_factor",
                "discharge_loss_factor",
                "pv_correction_factor",
            )
        },
        "capsule_id": capsule_ids[-1] if capsule_ids else None,
        "capsule_ids": capsule_ids,
        "dataset_version": batch.dataset_version,
        "chariot_model_version": batch.model_version,
        "residual_version": batch.residual_version,
        "residual_weight": batch.residual_weight,
        "scenario_batch": {
            "batch_id": batch.batch_id,
            "origin_utc": batch.origin_utc,
            "n_scenarios": batch.n_scenarios,
            "horizon_steps": batch.horizon_steps,
            "content_hash": batch.content_hash,
        },
        "tribunal_result": {
            "n_proposed": result.n_proposed,
            "n_retained": result.n_retained,
            "n_rejected": result.n_rejected,
            "survival_rate": round(float(result.survival_rate), 6),
            "constraint_counts": result.constraint_counts,
            "policy_version": runtime.tribunal_policy.policy_version,
            "solver_version": runtime.tribunal_policy.solver_version,
        },
        "confidence_report": {
            "report_id": report.report_id,
            "status": report.status,
            "confidence": round(float(report.confidence), 6),
            "policy_version": report.policy_version,
        },
        "policies": policy_snapshot(),
    }


def _generate_batch(runtime: Runtime, request: ScenarioRequest) -> dict[str, Any]:
    model = _model_for(runtime, request.model_version, request.site_id)
    ctx = runtime.context_for(request.tenant_id, request.site_id)
    frame = (
        _frame_from_records(request.records)
        if request.records
        else runtime.frame_for(request.tenant_id, request.site_id)
    )
    steps = min(HORIZONS.get(request.horizon, 96), len(frame))
    future = frame.iloc[:steps].reset_index(drop=True)
    # The generator and the Tribunal must see the *same* calibration, or the
    # energy balance law cannot hold (the Tribunal is fail-closed).
    factors = cal.resolve(
        runtime.store, tenant_id=request.tenant_id, site_id=request.site_id
    )
    batch = gen.generate_batch(
        model,
        request.site_id,
        future,
        tenant_id=request.tenant_id,
        n_scenarios=request.n_scenarios,
        seed=request.seed,
        ctx=ctx,
        calibration=factors.to_dict(),
        dt_h=0.25,
    )
    runtime.batches[batch.batch_id] = batch
    return {
        "batch_id": batch.batch_id,
        "origin_utc": batch.origin_utc,
        "n_scenarios": batch.n_scenarios,
        "horizon_steps": batch.horizon_steps,
        "content_hash": batch.content_hash,
    }


app = create_app(os.environ.get("CHARIOT_STORE", "var/chariot.db"))


def main() -> None:  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "chariot.api:app",
        host=os.environ.get("CHARIOT_HOST", "0.0.0.0"),
        port=int(os.environ.get("CHARIOT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
