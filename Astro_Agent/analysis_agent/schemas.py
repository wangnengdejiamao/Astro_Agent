"""Pydantic shared-context schema for the astronomy multi-agent platform.

The platform state must be JSON-serializable because it is exchanged among the
PaperOrchestra orchestrator, Codex compute wrapper, astro_toolbox runners, QA,
and the ApJ drafter. Physical quantities are therefore stored as explicit
``{"value": ..., "unit": ...}`` records and can be converted back to
``astropy.units.Quantity`` when numerical code needs them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisStage(str, Enum):
    INITIALIZED = "Initialized"
    DATA_INGESTION = "Data_Ingestion"
    KG_RETRIEVAL = "KG_Retrieval"
    RAG_RETRIEVAL = "RAG_Retrieval"
    MODEL_SELECTION = "Model_Selection"
    FITTING = "Fitting"
    RESIDUAL_REVIEW = "Residual_Review"
    SYSTEMATICS = "Systematics"
    QA_REVIEW = "QA_Review"
    HUMAN_REVIEW = "Human_Review"
    DRAFTING = "Drafting"
    PEER_REVIEW = "Peer_Review"
    FINALIZED = "Finalized"
    ABORTED = "Aborted"


class ReviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKER = "blocker"


class QuantityValue(BaseModel):
    """JSON-safe physical quantity with an astropy-compatible unit."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    value: float | List[float]
    unit: str = Field(..., description="Astropy-compatible unit string, e.g. K, dex, mas, pc, solMass")
    uncertainty: Optional[float | List[float]] = None
    uncertainty_unit: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None

    @field_validator("unit", "uncertainty_unit")
    @classmethod
    def validate_astropy_unit(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            return str(u.Unit(value))
        except Exception as exc:
            raise ValueError(f"Invalid astropy unit '{value}': {exc}") from exc

    def to_quantity(self) -> u.Quantity:
        return self.value * u.Unit(self.unit)

    def uncertainty_quantity(self) -> Optional[u.Quantity]:
        if self.uncertainty is None:
            return None
        return self.uncertainty * u.Unit(self.uncertainty_unit or self.unit)

    @classmethod
    def from_quantity(
        cls,
        quantity: u.Quantity,
        *,
        uncertainty: Optional[u.Quantity] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
    ) -> "QuantityValue":
        uncertainty_value = None
        uncertainty_unit = None
        if uncertainty is not None:
            uncertainty_value = uncertainty.value
            uncertainty_unit = str(uncertainty.unit)
        return cls(
            value=quantity.value,
            unit=str(quantity.unit),
            uncertainty=uncertainty_value,
            uncertainty_unit=uncertainty_unit,
            description=description,
            source=source,
        )


class J2000Coordinates(BaseModel):
    """Source coordinates in the J2000/ICRS convention."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ra: QuantityValue = Field(..., description="Right ascension; normally deg")
    dec: QuantityValue = Field(..., description="Declination; normally deg")
    frame: Literal["ICRS"] = "ICRS"
    equinox: Literal["J2000"] = "J2000"
    name_resolver: Optional[str] = None
    crossmatch_radius: QuantityValue = Field(default_factory=lambda: QuantityValue(value=3.0, unit="arcsec"))

    @model_validator(mode="after")
    def validate_ra_dec_ranges(self) -> "J2000Coordinates":
        ra_deg = self.ra.to_quantity().to_value(u.deg)
        dec_deg = self.dec.to_quantity().to_value(u.deg)
        if not 0.0 <= float(ra_deg) < 360.0:
            raise ValueError(f"RA must be in [0, 360) deg, got {ra_deg}")
        if not -90.0 <= float(dec_deg) <= 90.0:
            raise ValueError(f"Dec must be in [-90, 90] deg, got {dec_deg}")
        return self

    @classmethod
    def from_degrees(
        cls,
        ra_deg: float,
        dec_deg: float,
        *,
        name_resolver: Optional[str] = None,
        radius_arcsec: float = 3.0,
    ) -> "J2000Coordinates":
        return cls(
            ra=QuantityValue(value=ra_deg, unit="deg", description="J2000 right ascension"),
            dec=QuantityValue(value=dec_deg, unit="deg", description="J2000 declination"),
            name_resolver=name_resolver,
            crossmatch_radius=QuantityValue(value=radius_arcsec, unit="arcsec"),
        )


class KGExperience(BaseModel):
    """Historical experience retrieved from the local knowledge graph."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    subject: str
    relation: str
    object: str
    evidence: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    suggested_action: Optional[str] = None
    method_transfer_risk: ReviewSeverity = ReviewSeverity.INFO


class RAGEvidence(BaseModel):
    """Local literature hit used by the research and writing agents."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    bibcode: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    section: Optional[str] = None
    snippet: Optional[str] = None
    methods: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class ArtifactBundle(BaseModel):
    """Uniform handoff object for figures, tables, and LaTeX snippets."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    key: str = Field(..., description="Stable key consumed by Drafter, e.g. sed_plot_path")
    path: Optional[str] = Field(None, description="Filesystem path for a figure/table/data product")
    latex: Optional[str] = Field(None, description="LaTeX table or figure snippet")
    caption: Optional[str] = None
    produced_by: Optional[str] = None
    units_checked: bool = False
    qa_status: Literal["pending", "approved", "rejected"] = "pending"


class FittingAttempt(BaseModel):
    """One Codex/astro_toolbox fitting attempt."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    attempt_index: int
    instruction: str
    tool: str
    status: Literal["planned", "running", "success", "failed", "repaired"]
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    error_type: Optional[str] = None
    repair_summary: Optional[str] = None
    artifacts: List[ArtifactBundle] = Field(default_factory=list)


class FittingResult(BaseModel):
    """Physical-model output with explicit units and retry history."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    model_name: str
    status: Literal["not_started", "running", "converged", "nonconverged", "invalid", "needs_human"]
    physical_parameters: Dict[str, QuantityValue] = Field(default_factory=dict)
    priors: Dict[str, QuantityValue | str | float | int | bool] = Field(default_factory=dict)
    goodness_of_fit: Dict[str, float] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    attempts: List[FittingAttempt] = Field(default_factory=list)
    artifacts: List[ArtifactBundle] = Field(default_factory=list)


class ReviewComment(BaseModel):
    """QA, reviewer, or human-in-the-loop comment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    reviewer: str
    severity: ReviewSeverity
    stage: AnalysisStage
    message: str
    evidence_path: Optional[str] = None
    recommended_action: Optional[str] = None
    blocks_workflow: bool = False


class CodexJob(BaseModel):
    """Natural-language job submitted to Codex-as-compute-node."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    job_id: str
    natural_language_instruction: str
    target_tool: Optional[str] = None
    max_retries: int = 3
    retry_on: List[str] = Field(default_factory=lambda: ["ValueError"])
    status: Literal["queued", "running", "success", "failed", "needs_human"] = "queued"
    attempts: List[FittingAttempt] = Field(default_factory=list)
    returned_artifacts: List[ArtifactBundle] = Field(default_factory=list)


class SharedContext(BaseModel):
    """Global JSON state exchanged by all astronomy research agents."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)

    run_id: str
    target_name: str
    coordinates: J2000Coordinates
    current_stage: AnalysisStage = AnalysisStage.INITIALIZED

    kg_experiences: List[KGExperience] = Field(default_factory=list)
    rag_evidence: List[RAGEvidence] = Field(default_factory=list)
    fitting_results: Dict[str, FittingResult] = Field(default_factory=dict)
    review_comments: List[ReviewComment] = Field(default_factory=list)
    codex_jobs: List[CodexJob] = Field(default_factory=list)
    drafter_artifacts: Dict[str, ArtifactBundle] = Field(default_factory=dict)

    paper_orchestra_workspace: Optional[str] = None
    astrotool_output_root: Optional[str] = None
    abnormal_report_path: Optional[str] = None
    human_review_required: bool = False

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_review(
        self,
        *,
        reviewer: str,
        severity: ReviewSeverity,
        stage: AnalysisStage,
        message: str,
        evidence_path: Optional[str] = None,
        recommended_action: Optional[str] = None,
        blocks_workflow: bool = False,
    ) -> None:
        self.review_comments.append(
            ReviewComment(
                reviewer=reviewer,
                severity=severity,
                stage=stage,
                message=message,
                evidence_path=evidence_path,
                recommended_action=recommended_action,
                blocks_workflow=blocks_workflow,
            )
        )
        if blocks_workflow or severity in {ReviewSeverity.CRITICAL, ReviewSeverity.BLOCKER}:
            self.human_review_required = True
            self.current_stage = AnalysisStage.HUMAN_REVIEW

    def artifact_map_for_drafter(self) -> Dict[str, Dict[str, Any]]:
        """Return key-value artifacts consumed by the ApJ Drafter."""
        return {
            key: artifact.model_dump(mode="json", exclude_none=True)
            for key, artifact in self.drafter_artifacts.items()
        }


def example_context() -> SharedContext:
    """Small valid context useful for smoke tests and documentation."""

    return SharedContext(
        run_id="example-run",
        target_name="Example Source",
        coordinates=J2000Coordinates.from_degrees(ra_deg=232.3955, dec_deg=29.4672),
        current_stage=AnalysisStage.DATA_INGESTION,
        fitting_results={
            "sed": FittingResult(
                model_name="SED baseline",
                status="not_started",
                priors={
                    "dust_temperature": QuantityValue(value=[200.0, 600.0], unit="K"),
                },
            )
        },
        drafter_artifacts={
            "sed_plot_path": ArtifactBundle(
                key="sed_plot_path",
                path="./output/fig_sed.pdf",
                caption="Observed photometry and best-fit SED model.",
                produced_by="astro_toolbox.sed",
                units_checked=True,
            )
        },
    )
