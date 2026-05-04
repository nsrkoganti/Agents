"""
Central state object passed between all agents via LangGraph.
Every agent reads from and writes to this state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class PhysicsType(str, Enum):
    CFD_INCOMPRESSIBLE = "CFD_incompressible"
    CFD_INCOMPRESSIBLE_TURBULENT = "CFD_incompressible_turbulent"
    CFD_COMPRESSIBLE = "CFD_compressible"
    FEA_STATIC_LINEAR = "FEA_static_linear"
    FEA_STATIC_NONLINEAR = "FEA_static_nonlinear"
    FEA_DYNAMIC = "FEA_dynamic"
    THERMAL = "thermal"
    THERMAL_STRUCTURAL = "thermal_structural"
    MULTIPHYSICS = "multiphysics"


class MeshType(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED_POLY = "unstructured_polyhedral"
    UNSTRUCTURED_TET = "unstructured_tetrahedral"
    UNSTRUCTURED_HEX = "unstructured_hexahedral"
    POINT_CLOUD = "point_cloud"
    TABULAR = "tabular"


class ProblemType(str, Enum):
    SCALAR_REGRESSION = "scalar_regression"
    FIELD_REGRESSION = "field_regression"
    OPERATOR_LEARNING = "operator_learning"
    TIME_SERIES = "time_series"
    CLASSIFICATION = "classification"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProblemCard:
    """Output of Analyst Agent — describes the problem fully."""
    problem_type: ProblemType = ProblemType.FIELD_REGRESSION
    physics_type: PhysicsType = PhysicsType.CFD_INCOMPRESSIBLE_TURBULENT
    mesh_type: MeshType = MeshType.UNSTRUCTURED_POLY
    input_features: List[str] = field(default_factory=list)
    output_targets: List[str] = field(default_factory=list)
    data_size: int = 0
    n_nodes: int = 0
    n_cells: int = 0
    physics_constraints: List[str] = field(default_factory=list)
    turbulence_model: Optional[str] = None
    re_number: Optional[float] = None
    accuracy_threshold: Dict[str, float] = field(default_factory=lambda: {
        "r2": 0.92, "rel_l2": 0.05, "max_point_error": 0.15
    })
    special_flags: List[str] = field(default_factory=list)
    software_source: str = "STAR-CCM+"
    geometry_description: str = ""


@dataclass
class ModelCandidate:
    """A single ML model candidate with scores."""
    name: str = ""
    family: str = ""
    github_url: str = ""
    install_cmd: str = ""
    paper: str = ""
    mesh_requirement: str = "any"
    min_data_samples: int = 0
    supports_field_output: bool = True
    has_builtin_physics_loss: bool = False
    benchmark_l2_error: float = 0.05
    inference_speed_score: float = 5.0
    code_maturity_stars: int = 0
    # Scoring results
    scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    github_report: Dict[str, Any] = field(default_factory=dict)
    skip_reason: Optional[str] = None
    install_verified: bool = False


@dataclass
class TrainingResult:
    """Output of Trainer Agent."""
    model_name: str = ""
    model_object: Any = None
    train_loss: float = float('inf')
    val_loss: float = float('inf')
    training_epochs: int = 0
    training_time_seconds: float = 0.0
    converged: bool = False
    config_used: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Output of Evaluator Agent."""
    r2_score: float = 0.0
    rel_l2_error: float = 1.0
    max_point_error: float = 1.0
    inference_time_ms: float = 0.0
    passed: bool = False
    failure_reason: Optional[str] = None
    failure_diagnosis: Optional[str] = None
    recommended_fix: Optional[str] = None


@dataclass
class PhysicsReport:
    """Output of Physics Master Agent — aggregates all 5 sub-agents."""
    governing_equations_passed: bool = False
    governing_equations_detail: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions_passed: bool = False
    boundary_conditions_detail: Dict[str, Any] = field(default_factory=dict)
    conservation_passed: bool = False
    conservation_detail: Dict[str, Any] = field(default_factory=dict)
    turbulence_passed: bool = False
    turbulence_detail: Dict[str, Any] = field(default_factory=dict)
    material_passed: bool = False
    material_detail: Dict[str, Any] = field(default_factory=dict)
    overall_passed: bool = False
    fix_instructions: Optional[str] = None
    physics_lambda_updates: Dict[str, float] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Record of a single training attempt."""
    attempt_number: int = 0
    model_name: str = ""
    evaluation_result: Optional[EvaluationResult] = None
    physics_report: Optional[PhysicsReport] = None
    overall_passed: bool = False
    failure_reason: Optional[str] = None
    fix_applied: Optional[str] = None


@dataclass
class AgentSystemState:
    """
    Master state object — shared across ALL agents in LangGraph.
    Every agent reads and writes fields here.
    """
    # Input
    data_path: str = ""
    software_source: str = "STAR-CCM+"
    run_id: str = ""

    # Data Agent outputs
    dataset: Any = None
    unified_schema: Dict[str, Any] = field(default_factory=dict)
    data_agent_status: AgentStatus = AgentStatus.PENDING

    # Analyst Agent outputs
    problem_card: Optional[ProblemCard] = None
    analyst_status: AgentStatus = AgentStatus.PENDING

    # Selector Agent outputs
    all_candidates: List[ModelCandidate] = field(default_factory=list)
    ranked_shortlist: List[ModelCandidate] = field(default_factory=list)
    selected_model: Optional[ModelCandidate] = None
    selector_thinking_log: List[str] = field(default_factory=list)
    selector_status: AgentStatus = AgentStatus.PENDING

    # Trainer Agent outputs
    training_result: Optional[TrainingResult] = None
    trainer_status: AgentStatus = AgentStatus.PENDING
    physics_lambda_weights: Dict[str, float] = field(default_factory=lambda: {
        "continuity": 1.0, "momentum": 0.5, "bc": 2.0,
        "equilibrium": 1.0, "symmetry": 1.0
    })

    # Evaluator Agent outputs
    evaluation_result: Optional[EvaluationResult] = None
    evaluator_status: AgentStatus = AgentStatus.PENDING

    # Physics Agent outputs
    physics_report: Optional[PhysicsReport] = None
    physics_status: AgentStatus = AgentStatus.PENDING

    # Iteration tracking
    iteration_log: List[IterationRecord] = field(default_factory=list)
    current_attempt: int = 0
    current_model_index: int = 0
    iteration_status: AgentStatus = AgentStatus.PENDING

    # Verifier Agent outputs
    verification_passed: bool = False
    verification_detail: Dict[str, Any] = field(default_factory=dict)
    verifier_status: AgentStatus = AgentStatus.PENDING

    # Saver Agent outputs
    saved_model_path: str = ""
    metadata_path: str = ""
    saver_status: AgentStatus = AgentStatus.PENDING

    # Global status
    pipeline_complete: bool = False
    pipeline_success: bool = False
    error_message: Optional[str] = None
    thinking_log: List[str] = field(default_factory=list)

    # Self-learning
    knowledge_recommendation: Optional[str] = None
    similar_problem_found: bool = False
    custom_model_attempted: bool = False

    # Architect
    architect_triggered: bool = False
    custom_model_dna: Optional[dict] = None
    custom_model_code: Optional[str] = None
