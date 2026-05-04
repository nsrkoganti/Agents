"""Rule-based problem classifier — backup to LLM classification."""

from agents.orchestrator.agent_state import PhysicsType, MeshType, ProblemType


class ProblemClassifier:
    """Classifies simulation problems based on field names and metadata."""

    CFD_FIELDS = {"velocity", "pressure", "k", "omega", "epsilon", "yplus", "temperature"}
    FEA_FIELDS = {"stress", "strain", "displacement", "von_mises", "force", "reaction"}

    def classify_physics(self, field_names: list) -> PhysicsType:
        names_lower = {n.lower() for n in field_names}
        cfd_score = len(names_lower & self.CFD_FIELDS)
        fea_score = len(names_lower & self.FEA_FIELDS)

        if cfd_score > fea_score:
            if any(f in names_lower for f in ["k", "omega", "epsilon"]):
                return PhysicsType.CFD_INCOMPRESSIBLE_TURBULENT
            return PhysicsType.CFD_INCOMPRESSIBLE
        elif fea_score > 0:
            return PhysicsType.FEA_STATIC_LINEAR
        return PhysicsType.CFD_INCOMPRESSIBLE_TURBULENT

    def classify_mesh(self, mesh_type_str: str) -> MeshType:
        mapping = {
            "structured":    MeshType.STRUCTURED,
            "poly":          MeshType.UNSTRUCTURED_POLY,
            "polyhedral":    MeshType.UNSTRUCTURED_POLY,
            "tet":           MeshType.UNSTRUCTURED_TET,
            "tetrahedral":   MeshType.UNSTRUCTURED_TET,
            "hex":           MeshType.UNSTRUCTURED_HEX,
            "hexahedral":    MeshType.UNSTRUCTURED_HEX,
            "point_cloud":   MeshType.POINT_CLOUD,
            "tabular":       MeshType.TABULAR,
        }
        for key, val in mapping.items():
            if key in mesh_type_str.lower():
                return val
        return MeshType.UNSTRUCTURED_POLY
