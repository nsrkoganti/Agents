from models.architectures.mlp_surrogate import MLPSurrogate
from models.architectures.transolver import TransolverSurrogate
from models.architectures.pinn import PINNSurrogate
from models.architectures.fno_surrogate import FNOSurrogate
from models.architectures.gnn_surrogate import GNNSurrogate
from models.architectures.hybrid_model import HybridTransolverPINN

__all__ = [
    "MLPSurrogate",
    "TransolverSurrogate",
    "PINNSurrogate",
    "FNOSurrogate",
    "GNNSurrogate",
    "HybridTransolverPINN",
]
