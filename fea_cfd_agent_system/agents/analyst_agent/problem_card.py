"""Problem card builder — constructs ProblemCard from classified data."""

from agents.orchestrator.agent_state import ProblemCard, PhysicsType, MeshType, ProblemType


def build_problem_card(
    physics_type: PhysicsType,
    mesh_type: MeshType,
    n_cases: int,
    n_nodes: int,
    n_cells: int,
    software: str,
    fields: list = None,
    turbulence_model: str = None,
    re_number: float = None,
) -> ProblemCard:
    """Factory function to build a ProblemCard."""
    return ProblemCard(
        problem_type=ProblemType.FIELD_REGRESSION,
        physics_type=physics_type,
        mesh_type=mesh_type,
        data_size=n_cases,
        n_nodes=n_nodes,
        n_cells=n_cells,
        software_source=software,
        turbulence_model=turbulence_model,
        re_number=re_number,
        input_features=["geometry", "boundary_conditions"],
        output_targets=fields or ["velocity_field", "pressure_field"],
    )
