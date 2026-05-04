# FEA & CFD Autonomous ML Agent System

Autonomous AI pipeline that ingests simulation data, thinks about the problem,
selects the best ML model, enforces physics laws, and saves a verified surrogate model.

## Quick Start

```bash
# Install
pip install -e .

# Set API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and GITHUB_TOKEN

# Run on STAR-CCM+ export
python main.py --data ./data/raw/my_duct_export/ --software STAR-CCM+

# Run arXiv discovery
python main.py --data ./data/raw/ --discover

# Run tests
pytest tests/ -v
```

## Agent Pipeline

```
Data → Analyst → Selector → Trainer → Evaluator → Physics Agent → Iteration → Verifier → Saver → Self-Learning Update
```

## Physics Agent Sub-Agents

- Governing Equation Agent (Navier-Stokes, FEA equilibrium)
- Boundary Condition Agent (no-slip, inlet, outlet)
- Conservation Agent (mass, energy)
- Turbulence Agent (k-ω SST, Re, y⁺)
- Material Agent (elasticity, damage)

## Self-Learning

Every run is stored in a permanent SQLite database. The system learns:
- Which models work best for each physics/mesh/data-size combination
- Which lambda weights fix which physics violations
- New architectures from arXiv (weekly discovery)
- Custom models it designed itself

## Custom Model Architect

When all known models fail (after 12 attempts), the Architect Agent:
1. Analyzes all failures with LLM
2. Designs a new architecture DNA (Transolver/PINN/GNN/Hybrid)
3. Uses Neural Architecture Search (Optuna) to optimize hyperparameters
4. Generates complete PyTorch code with LLM
5. Validates and trains the custom model

## Supported Software

STAR-CCM+, ANSYS Fluent, OpenFOAM, Abaqus

## Target Geometry (First Test Case)

Rectangular duct, 8mm wall, 15mm and 20mm radii corners, k-ω SST turbulence

## Architecture Block Library

- **PhysicsAttentionBlock**: Transolver-style (Wu et al. ICML 2024) — physics-state slicing
- **FourierLayer**: FNO-style spectral convolution
- **GraphConvBlock**: MeshGraphNet-style message passing
- **CoordinateEmbedding**: Fourier feature mapping for mesh coordinates
- **BoundaryConditionEncoder**: Encodes BC type + value as learned embeddings
