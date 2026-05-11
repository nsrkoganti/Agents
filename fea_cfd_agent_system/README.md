# FEA Surrogate ML Agent System

An autonomous multi-agent system that ingests Finite Element Analysis (FEA) simulation data from any major solver, automatically selects and trains a best-fit ML surrogate model, enforces physical consistency, and learns from every run to improve future decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agent Pipeline](#agent-pipeline)
4. [ML Model Architectures](#ml-model-architectures)
5. [Physics Validation](#physics-validation)
6. [RAG & Self-Learning](#rag--self-learning)
7. [Data Ingestion](#data-ingestion)
8. [LLM Configuration](#llm-configuration)
9. [Custom Architect](#custom-architect)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Configuration](#configuration)
13. [Testing](#testing)
14. [Project Structure](#project-structure)

---

## Overview

FEA solvers (ANSYS, Abaqus, CalculiX, STAR-CCM+) produce high-fidelity results but are slow — hours to days per simulation. This system trains a surrogate ML model that reproduces those results in milliseconds, while guaranteeing that predictions satisfy the underlying physics laws (equilibrium, constitutive relations, strain compatibility, boundary conditions).

**Key capabilities:**

- Reads `.rst`, `.odb`, `.frd`, `.vtu`, `.h5`, `.csv`, `.npz`, `.npy` files directly
- Selects among 13 research-grade model architectures automatically
- Enforces 5 FEA physics checks during training via soft-penalty loss
- Falls back to LLM-designed novel architectures when standard models fail
- Retrieves relevant past experiences (RAG) to improve iteration decisions
- Persists all runs in SQLite + FAISS for compounding self-improvement

**Typical results:** R² ≥ 0.95, Relative L2 ≤ 3%, inference < 100 ms for FEA_static_linear problems.

---

## Architecture

```
Input (FEA result file  OR  --search-datasets flag)
         │
    ┌────▼──────────────────────────────────────────────────┐
    │              LangGraph StateGraph                      │
    │                                                        │
    │  DataAgent → AnalystAgent → SelectorAgent             │
    │      │                          │                     │
    │  [solver loaders]         [model registry +           │
    │  [NumpyLoader]             GitHub/arXiv discovery]    │
    │                                 │                     │
    │             [DatasetOrchestrator]                     │
    │             (if --search-datasets)                    │
    │                                 │                     │
    │         TrainerAgent ←──────────┘                    │
    │              │                                        │
    │         EvaluatorAgent ──→ FAIL → IterationAgent     │
    │              │                         │              │
    │           PASS                  retry / select_new /  │
    │              │                  ArchitectAgent        │
    │         PhysicsMasterAgent                            │
    │              │                                        │
    │         FAIL → IterationAgent (lambda update)         │
    │              │                                        │
    │           PASS                                        │
    │              │                                        │
    │         VerifierAgent (Claude Opus 4.7)               │
    │              │                                        │
    │         SaverAgent → SelfLearningUpdater              │
    └────────────────────────────────────────────────────────┘
```

All agents read/write a single `AgentSystemState` dataclass. Routing is conditional — `EvaluatorAgent` routes to Physics on pass, `IterationAgent` on fail. `PhysicsMasterAgent` routes to Verifier on pass, `IterationAgent` (with updated lambdas) on fail. Maximum 24 total attempts.

---

## Agent Pipeline

### DataAgent
Detects file format from extension, dispatches to the correct solver loader, runs quality inspection (NaN/Inf check, mesh skewness), and assembles a unified schema.

**Format dispatch:**

| Extension | Loader | Library |
|---|---|---|
| `.rst`, `.rth` | ANSYSLoader | ansys-mapdl-reader |
| `.odb`, `.inp` | AbaqusLoader | abapy, meshio |
| `.frd`, `.dat` | CalculiXLoader | ccx2paraview, meshio |
| `.csv` (STAR-CCM+) | StarCCMFEALoader | pandas |
| `.vtu`, `.vtk`, `.h5` | VTKLoader / HDF5Loader | pyvista, h5py |
| `.npz`, `.npy` | NumpyLoader | numpy |

**Unified schema output:**
```python
{
    "nodes":           np.ndarray,   # (N, 3) — xyz
    "elements":        np.ndarray,   # (E, K) — connectivity
    "fields": {
        "displacement": np.ndarray,  # (N, 3)
        "stress":       np.ndarray,  # (N, 6) — Voigt
        "strain":       np.ndarray,  # (N, 6) — Voigt
        "von_mises":    np.ndarray,  # (N,)
        "temperature":  np.ndarray,  # (N,) — if thermal
    },
    "n_nodes":         int,
    "n_elements":      int,
    "physics_type":    str,          # "FEA_static_linear" | ...
    "solver_source":   str,          # "ANSYS" | "Abaqus" | ...
    "mesh_type":       str,
    "element_type":    str,          # "tet" | "hex" | "shell" | "beam"
    "boundary_info":   dict,
    "material_properties": dict,
    "load_steps":      int,
}
```

**NumpyLoader** supports three batch layouts automatically:
- Flat single-case: `nodes(N,3)`, `displacement(N,3)`
- Stacked batch: `nodes(C,N,3)`, `displacement(C,N,3)` (3D arrays → C cases)
- Object array: `cases` key holding array of dicts

---

### AnalystAgent
Reads the unified schema and produces a `ProblemCard`:

```python
@dataclass
class ProblemCard:
    physics_type:     PhysicsType    # FEA_static_linear | FEA_static_nonlinear | FEA_dynamic | thermal | ...
    mesh_type:        MeshType       # unstructured_tetrahedral | structured | ...
    material_model:   str            # linear_elastic | hyperelastic | elastoplastic
    loading_type:     str            # static | dynamic | thermal | cyclic
    element_type:     str            # tet | hex | shell | beam
    n_load_steps:     int
    solver_source:    str
    yield_stress:     Optional[float]
    accuracy_threshold: Dict[str, float]   # {"r2": 0.95, "rel_l2": 0.03, ...}
    geometry_description: str
```

Auto-classification rules:

| Signal | Classification |
|---|---|
| Time arrays in data | `loading_type = "dynamic"` |
| Temperature field | `loading_type = "thermal"` |
| Multiple load steps | `loading_type = "cyclic"` |
| `yield_stress` in material | `material_model = "elastoplastic"` |
| 4 nodes/element | `element_type = "tet"` |
| 8 nodes/element | `element_type = "hex"` |

Accuracy defaults:
- `FEA_static_linear`: R² ≥ 0.95, Rel L2 ≤ 3%
- `FEA_static_nonlinear`: R² ≥ 0.93, Rel L2 ≤ 5%
- `FEA_dynamic`: R² ≥ 0.90, Rel L2 ≤ 8%
- `thermal`: R² ≥ 0.95, Rel L2 ≤ 3%

---

### SelectorAgent (DeepThinkingSelector)
Scores all 13 registered models against the `ProblemCard` using weighted criteria:

```
total_score = 0.25 × physics_fit
            + 0.20 × mesh_fit
            + 0.15 × data_efficiency
            + 0.15 × accuracy
            + 0.10 × physics_law_support
            + 0.05 × speed
            + 0.05 × code_maturity
            + 0.05 × compute_cost
```

RAG augments scoring: retrieves the top-8 most similar past runs per candidate and computes dynamic success rate + average R² for this specific physics/mesh combination — overriding the static table scores.

Also runs `DiscoveryAgent` to scan GitHub and arXiv for newer architectures not yet in the registry.

---

### TrainerAgent
Trains the selected model with a composite physics-informed loss:

```
L_total = w_data × MSE(ŷ, y)
        + λ_eq   × ‖F_int − F_ext‖ / ‖F_ext‖
        + λ_bc   × ‖u_bc − u_prescribed‖²
        + λ_sym  × ‖σ − σᵀ‖_F
        + λ_comp × ‖ε − sym(∇u)‖_F
        + λ_con  × ‖σ − C:ε‖_F / ‖σ‖
```

Training defaults: up to 2000 epochs, AdamW, early stopping (patience=20), gradient clipping (max_norm=1.0). `AutoConfigurator` selects batch size, learning rate, and scheduler based on dataset size and physics type.

---

### EvaluatorAgent
Checks three metrics against thresholds from `ProblemCard.accuracy_threshold`:

| Metric | Formula | Default threshold |
|---|---|---|
| R² | `1 − SSres/SStot` | ≥ 0.92 |
| Relative L2 | `‖ŷ − y‖ / ‖y‖` | ≤ 0.05 |
| Max point error | `max|ŷᵢ − yᵢ| / max|yᵢ|` | ≤ 0.15 |
| Inference time | milliseconds per prediction | ≤ 100 ms |

Routes to PhysicsMasterAgent on pass, IterationAgent on fail.

---

### IterationAgent
On failure, calls the dev LLM with:
- Current metrics vs thresholds
- RAG-retrieved similar past failures + the fix that worked
- Physics check results (which of 5 checks failed)
- Attempt count

LLM selects one of these fix strategies:
- `increase_epochs` / `reduce_lr` / `increase_batch`
- `update_lambda` (multiply failing penalty weight)
- `re_encode_bc` (change how boundary nodes are encoded)
- `select_new_model` (skip to next candidate)
- `trigger_architect` (after 12 failures, escalate)

---

### VerifierAgent
Uses Claude Opus 4.7 to issue a physics certificate:

```json
{
  "equilibrium":         {"passed": true,  "detail": {...}},
  "stress_strain":       {"passed": true,  "detail": {...}},
  "compatibility":       {"passed": true,  "detail": {...}},
  "boundary_conditions": {"passed": true,  "detail": {...}},
  "material":            {"passed": true,  "detail": {...}},
  "overall":             "PHYSICALLY VERIFIED",
  "certified_at":        "2026-05-11T14:23:01"
}
```

---

### SaverAgent + SelfLearningUpdater
Saves the validated model to `models/saved/` with full metadata. Updates:
- `RunDatabase` (SQLite) — run record, metrics, physics certificate
- RAG indices — geometry, failure, physics_model, lambda, custom_dna
- `PerformanceTracker` — per-model success rate, average R², failure counts
- `KnowledgeBase` — lambda weight history for this physics type

---

## ML Model Architectures

All models inherit `BaseSurrogateModel` with a common interface:
- **Input**: `(B, N, in_dim)` — batch of node feature sets
- **Output**: `(B, N, out_dim)` — batch of per-node field predictions

| Model | File | Best for | Complexity |
|---|---|---|---|
| **Transolver-3** | `transolver_3.py` | Industrial FEA, 1M+ nodes | O(N·S), S=64 slices |
| **Transolver++** | `transolver_pp.py` | General FEA, eidetic state persistence | O(N·S) |
| **EAGNN** | `eagnn.py` | Unstructured tet, nonlinear stress | O(N + E_aug) |
| **MeshGraphNet-Transformer** | `mgn_transformer.py` | Plasticity, large deformation | O(N + E) |
| **GNSS** | `gnss.py` | Structural dynamics, local frames | O(N + E) |
| **GS-PI-DeepONet** | `deeponet.py` | Small data (<100 samples) | O(N·P) |
| **Factorized FNO** | `fno_surrogate.py` | Structured grids | O(N log N) |
| **MeshGraphNet** | `gnn_surrogate.py` | General unstructured baseline | O(N + E) |
| **PINN** | `pinn.py` | Physics-constrained, very small data | O(N) |
| **MLP** | `mlp_surrogate.py` | Tabular / low-DOF baseline | O(N) |
| **Hybrid** | `hybrid_model.py` | Combines GNN + operator | varies |
| **MoE Surrogate** | `moe_surrogate.py` | Spatially heterogeneous fields | O(K × N) |
| **Transolver (orig.)** | `transolver.py` | Original ICML 2024 baseline | O(N·S) |

### Mixture of Experts (MoE)
`MixtureOfExpertsSurrogate` runs K expert models in parallel and combines them with per-node learned gate weights:

```
gate: (B, N, in_dim) → (B, N, K)   softmax
output = Σₖ gateₖ × expertₖ(x)
```

`NodeGatingNetwork` is a 2-layer MLP producing per-node soft weights. A load-balance auxiliary loss (weight=0.01) prevents expert collapse. `get_expert_attribution(x)` returns the spatial gate heatmap `(B, N, K)`.

---

## Physics Validation

Five agents run in parallel via `ThreadPoolExecutor` in `PhysicsMasterAgent`:

### EquilibriumAgent
Recovers internal nodal forces via virtual work and checks global balance:

```
‖F_int − F_ext‖ / ‖F_ext‖ < threshold
```

Threshold: 1×10⁻⁵ (static linear), 1×10⁻⁴ (static nonlinear), 1×10⁻⁴ (dynamic).

### StressStrainAgent
Three checks:
1. **Constitutive law**: `‖σ − C:ε‖ / ‖σ‖ < 0.02`
2. **Von Mises yield**: `σ_vm ≤ yield_stress` (elastoplastic only)
3. **Tensor symmetry**: `max|σᵢⱼ − σⱼᵢ| < 1×10⁻⁸`

### CompatibilityAgent
Strain-displacement consistency via finite differences:

```
‖ε − sym(∇u)‖ / ‖ε‖ < 1×10⁻⁷
```

Uses `scipy.spatial.cKDTree` to find nearest neighbors for the gradient stencil.

### BoundaryConditionAgent
- **Fixed supports**: `‖u‖ < 1×10⁻⁶` at constrained nodes
- **Applied loads**: reaction forces balance applied loads (Newton's 3rd law)
- **Symmetry planes**: normal displacement ≈ 0
- **Contact**: penetration = 0, contact pressure ≥ 0

### MaterialAgent
Validates the constitutive model:
- **Linear elastic**: Hooke's law `σ = C:ε`
- **Neo-Hookean**: `W = μ/2(I₁−3) − μln(J) + λ/2·ln²(J)`
- **J2 Elastoplastic**: von Mises yield + isotropic hardening `σ_y(εₚ) = σ_y0 + Hεₚ`
- **Damage**: `D ∈ [0,1]`, `σ_eff = σ/(1−D)`

### Lambda Updates
When a check fails, `physics_lambda_updates` multiplies the corresponding penalty weight before the next training attempt:

| Check | Multiplier |
|---|---|
| equilibrium | 2.0× |
| bc | 5.0× |
| symmetry | 2.0× |
| compatibility | 2.0× |
| constitutive | 2.0× |

---

## RAG & Self-Learning

### RAG Layer (`agents/rag/`)
Built on FAISS `IndexFlatIP` (inner product = cosine on L2-normalised vectors) with `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, ~90 MB, downloaded once).

Five named indices, each persisted to `memory/rag_indices/{name}.faiss` + `.meta.pkl`:

| Index | What is embedded | Used by |
|---|---|---|
| `geometry_index` | physics_type + mesh_type + geometry_description | SelectorAgent, PatternRecognizer |
| `failure_index` | failure_reason + model_name + fix_tried + outcome | IterationAgent |
| `physics_model_index` | model + physics_type + R² result | SelectorAgent scoring |
| `lambda_index` | physics_type + failed_checks + lambda_json that worked | KnowledgeBase |
| `custom_dna_index` | DNA block sequence + physics failure signature + R² | ArchitectAgent, CodeGenerator |

`RAGRetriever` is created once in `MasterOrchestrator.__init__()` and injected into all agents. It gracefully no-ops (logs a warning) if `faiss-cpu` or `sentence-transformers` are not installed.

**Five query methods:**
- `find_similar_problems(problem_card, top_k=5)` → past runs with outcomes
- `find_fixes_for_failure(model, failure_reason, physics_type, top_k=5)` → past fixes
- `find_model_history(model_name, problem_card, top_k=8)` → per-model performance history
- `find_lambda_history(physics_type, failed_checks, top_k=3)` → successful lambda configs
- `find_similar_custom_dna(physics_type, failed_checks, top_k=3)` → custom model DNA blocks

### Self-Learning (`agents/self_learning/`)
- **`RunDatabase`** — SQLite at `memory/experience.db`. Tables: `runs`, `failures`, `model_performance`, `lambda_history`, `custom_models`, `patterns`.
- **`KnowledgeBase`** — `recommend_lambda_weights()` queries RAG first (top-3 similar lambda configs), falls back to SQL max-R² record.
- **`PatternRecognizer`** — replaces SQL `LIKE` geometry queries with `find_similar_problems()` (cosine similarity ≥ 0.65 threshold).
- **`SelfLearningUpdater`** — runs after every pipeline completion; updates performance stats, increments model failure counts, indexes new run to all RAG indices.
- **`PerformanceTracker`** — per-model success rate, failure streak, and physics pass rate used by SelectorAgent to deprioritize repeatedly failing models.

Every `RunDatabase.save_run()`, `save_failure()`, and `save_custom_model()` also indexes into the appropriate RAG vector store — so the system gets smarter with every run automatically.

---

## Data Ingestion

### Supported Solvers

**ANSYS** (`.rst`, `.rth`)
Uses `ansys-mapdl-reader` for binary result files. Extracts UX/UY/UZ → displacement, SX/SY/SZ/SXY → stress Voigt, SEQV → von_mises, RF → reaction forces. Falls back to pyvista for VTK-exported results.

**Abaqus** (`.odb`, `.inp`)
Primary: `abapy` for ODB post-processing — reads `FieldOutput` objects (U, S, E, MISES, RF). Fallback: `meshio` for input deck geometry.

**CalculiX** (`.frd`, `.dat`)
Uses `meshio.read(".frd")` or `ccx2paraview` → `.vtu` → pyvista. Maps CalculiX field names to unified schema aliases.

**STAR-CCM+** (`.csv`)
No Python API — expects CSV or VTK export. Detects FEA fields by alias matching (`FEA_FIELD_ALIASES` in `field_extractor.py`). Works with both tabular (CSV) and VTK mesh exports.

**NumPy** (`.npz`, `.npy`)
Three batch layouts detected automatically:
1. Single-case flat: `nodes(N,3)`, `displacement(N,3)`
2. Stacked batch `(C,N,D)`: 3D arrays → exploded into C individual cases
3. Object array: `cases` key storing an array of dicts

Per-case 1D arrays (e.g. `applied_stress(C,)`, `E(C,)`, `nu(C,)`) are correctly sliced per case index.

### Field Aliases
`FieldExtractor` resolves solver-specific names to canonical names:

| Canonical | Aliases |
|---|---|
| `displacement` | U, DISP, UX_UY_UZ, deformation |
| `stress` | STRESS, S, sigma, stress_voigt |
| `von_mises` | vonMises, VM_STRESS, SEQV, Mises, S_vm |
| `strain` | STRAIN, E, epsilon, strain_voigt |
| `temperature` | T, TEMP, NT11 |
| `reaction` | RF, reaction_forces, REACTION_FORCE |

Von Mises is computed from Voigt stress tensor automatically if missing:
```
σ_vm = √(½[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)² + 6(σxy² + σyz² + σxz²)])
```

---

## LLM Configuration

Two distinct LLM roles — never mixed:

| Role | Function | Model | Fallback |
|---|---|---|---|
| **Dev LLM** | Analysis, ranking, iteration diagnosis, dataset search | Gemini 2.5 Flash (free: 15 RPM, 1500 RPD) | Claude Sonnet 4.6 |
| **Verifier LLM** | Code generation, physics certificate | Claude Opus 4.7 | — |

```python
# agents/shared/llm_factory.py
get_dev_llm()       # Gemini 2.5 Flash if GEMINI_API_KEY set, else Claude Sonnet 4.6
get_verifier_llm()  # Always Claude Opus 4.7 via ANTHROPIC_API_KEY
```

Set environment variables:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # required (verifier + dev fallback)
export GEMINI_API_KEY=AIza...         # optional — preferred for dev LLM (free)
```

---

## Custom Architect

Triggered automatically after 12 consecutive model failures. Seven-step flow:

```
1. _analyze_failures()
   → Dev LLM reads all IterationRecords, produces structured failure diagnosis
   → Identifies: which physics checks consistently fail, data size constraints,
     mesh type issues, training instability patterns

2. _choose_template()
   → Selects from 5 DNA templates: pinn | transolver | gnn | operator | hybrid
   → Tries each once before escalating to novel design

3. (After all 5 templates fail) _design_novel_architecture()
   → Verifier LLM (Claude Opus 4.7) reads failure analysis + available BlockTypes
   → Returns JSON block sequence: [{type, hidden_dim, ...}, ...]
   → ArchitectureDNA.from_llm_json() validates and constructs DNA object

4. NASEngine.refine_dna()  [Optuna, 30 trials]
   → Searches: block types, hidden_dim, n_layers, n_heads, dropout, residual

5. CodeGenerator.generate()
   → Verifier LLM writes complete PyTorch nn.Module from DNA
   → RAG injects similar working code as few-shot example

6. CodeGenerator.validate()
   → AST parse → import → forward pass → NaN check → gradient flow check
   → Weight norm range check (detects explosion)

7. Save to SQLite custom_models table
   → Future runs: get_best_custom_model(physics_type) reuses without retraining
   → RAG custom_dna_index: future novel designs build on proven block sequences
```

### Available Block Types (`architecture_dna.py` — `BlockType` enum)

| Block | Implementation | Use case |
|---|---|---|
| `PHYSICS_ATTN` | Physics-slice attention (Transolver-style) | General FEA |
| `FOURIER` | Spectral Fourier layer (FNO-style) | Structured grids |
| `GRAPH_CONV` | Message passing (PyG) | Unstructured mesh |
| `MLP_MIXER` | Token-mixing MLP | Fast baseline |
| `MAMBA_BLOCK` | Selective state-space scan, O(N) | Long-range, transient |
| `CONV_NEXT_BLOCK` | Depthwise Conv1d + LayerNorm | Mesh feature extraction |
| `CROSS_ATTENTION` | Cross-attention mesh↔physics tokens | Boundary coupling |
| `SPECTRAL_NORM` | Lipschitz-constrained linear block | Stability guarantee |

PyTorch implementations in `agents/model_architect/physics_block_library.py`.

---

## Installation

```bash
# Clone
git clone https://github.com/nsrkoganti/Agents.git
cd Agents/fea_cfd_agent_system

# Install (editable)
pip install -e .

# Core extras (choose what you need)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse
pip install neuraloperator deepxde
pip install sentence-transformers faiss-cpu   # RAG layer
pip install ansys-mapdl-reader abapy          # ANSYS / Abaqus loaders
pip install meshio pyvista ccx2paraview        # CalculiX / VTK loaders
pip install langchain-google-genai             # Gemini dev LLM (optional)

# Environment
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...          # optional
```

**Python requirement**: 3.10+

**GPU**: CUDA 12.1+ recommended for large meshes (>50k nodes). CPU mode works for development.

---

## Usage

### Run with existing FEA data

```bash
cd fea_cfd_agent_system

# ANSYS result file
python main.py --data path/to/result.rst --solver ansys --physics FEA_static_linear

# Abaqus ODB
python main.py --data result.odb --solver abaqus --physics FEA_static_nonlinear

# CalculiX
python main.py --data result.frd --solver calculix --physics FEA_dynamic

# NumPy batch file (200 cases)
python main.py --data data/synthetic_fea/plate_with_hole.npz --solver auto --physics FEA_static_linear

# VTK / HDF5
python main.py --data mesh_results.vtu --solver vtk --physics thermal
```

### Dataset search mode (no data file)

```bash
# Search HuggingFace, Zenodo, and curated list for FEA datasets
python main.py --search-datasets --physics FEA_static_nonlinear
```

### Generate synthetic data and smoke test

```bash
# Generate Kirsch plate-with-hole FEA dataset (200 cases, ~15 MB)
python scripts/generate_synthetic_fea_data.py --cases 200 --out data/synthetic_fea/plate_with_hole.npz

# Run end-to-end pipeline test (achieves R²≈0.99 with SCF-normalized target)
python scripts/test_synthetic_pipeline.py --cases 200 --epochs 300 --target von_mises
```

### Common options

```
--data           Path to FEA result file or directory
--solver         ansys | abaqus | calculix | starccm | vtk | hdf5 | auto
--physics        FEA_static_linear | FEA_static_nonlinear | FEA_dynamic | thermal | thermal_structural | multiphysics
--max-attempts   Override total iteration limit (default: 24)
--output-dir     Where to write model + certificate (default: results/)
--log-dir        Log file directory (default: logs/)
--db-path        SQLite experience database path (default: memory/experience.db)
--search-datasets  Enable dataset discovery mode (no data file required)
```

---

## Configuration

Configuration is loaded from two YAML files merged at startup:
1. `configs/base_config.yaml` — system, LLM, thresholds, training, physics
2. `configs/rag_config.yaml` — RAG enabled flag, indices, similarity threshold

Key sections:

```yaml
# Physics accuracy thresholds per FEA type
physics:
  fea:
    static_linear:
      r2_min: 0.95
      rel_l2_max: 0.03
      equilibrium_residual_max: 1.0e-5
    static_nonlinear:
      r2_min: 0.93
      rel_l2_max: 0.05

# Soft physics penalty weights (multiplied when check fails)
physics_lambda_weights:
  equilibrium: 1.0
  bc: 2.0
  compatibility: 1.0
  constitutive: 1.0
  symmetry: 1.0

lambda_multipliers:
  equilibrium: 2.0
  bc: 5.0      # BC violations penalized hardest

# Iteration limits
iteration:
  max_attempts_per_model: 3
  max_models_to_try: 8
  total_max_attempts: 24
```

Model-specific hyperparameters, GitHub URLs, and benchmark scores are in `configs/model_registry.yaml`.

---

## Testing

```bash
cd fea_cfd_agent_system

# All tests
pytest tests/ -v

# By layer
pytest tests/test_rag.py -v                 # RAG: embedding, FAISS add/search, persistence
pytest tests/test_fea_physics_agents.py -v  # 5 FEA physics checks
pytest tests/test_fea_data_loaders.py -v    # Solver loaders + NumpyLoader
pytest tests/test_moe_surrogate.py -v       # MoE: gates sum=1, shapes, load balance
pytest tests/test_novel_architect.py -v     # ArchitectureDNA, NAS, code validation
pytest tests/test_fea_models.py -v          # All 13 architecture forward passes
pytest tests/test_self_learning.py -v       # KnowledgeBase, PatternRecognizer, RunDatabase
pytest tests/test_full_pipeline.py -v       # End-to-end with mock data
```

**Smoke tests:**
```bash
# RAG round-trip
python -c "
from agents.rag.embedding_service import EmbeddingService
from agents.rag.vector_store import RAGVectorStore
svc = EmbeddingService()
store = RAGVectorStore('memory/rag_indices')
vec = svc.embed('FEA_static_linear unstructured_tet elastoplastic')
store.add('geometry_index', vec, {'run_id': 'test', 'r2': 0.95})
results = store.search('geometry_index', vec, top_k=1)
assert results[0]['metadata']['run_id'] == 'test'
print('RAG OK')
"

# MoE forward pass
python -c "
import torch
from models.architectures.moe_surrogate import MixtureOfExpertsSurrogate
from models.architectures.mlp_surrogate import MLPSurrogate
experts = [MLPSurrogate(in_dim=3, out_dim=6) for _ in range(3)]
moe = MixtureOfExpertsSurrogate(experts, in_dim=3, out_dim=6)
x = torch.randn(2, 64, 3)
out = moe(x)
assert out.shape == (2, 64, 6)
attr = moe.get_expert_attribution(x)
assert torch.allclose(attr.sum(-1), torch.ones(2, 64), atol=1e-5)
print('MoE OK')
"
```

---

## Project Structure

```
fea_cfd_agent_system/
├── main.py                         # CLI entry point; load_config() merges base + rag configs
├── setup.py
├── requirements.txt
│
├── agents/
│   ├── orchestrator/
│   │   ├── master_orchestrator.py  # LangGraph graph builder; initializes RAGRetriever
│   │   └── agent_state.py          # AgentSystemState, ProblemCard, PhysicsReport, enums
│   ├── data_agent/
│   │   ├── data_agent.py           # Format detection, quality inspection, schema assembly
│   │   ├── numpy_loader.py         # .npz/.npy — flat / stacked-batch / object-array layouts
│   │   ├── ansys_loader.py         # .rst via ansys-mapdl-reader
│   │   ├── abaqus_loader.py        # .odb via abapy; .inp via meshio
│   │   ├── calculix_loader.py      # .frd via ccx2paraview / meshio
│   │   ├── starccm_fea_loader.py   # CSV / VTK exports from STAR-CCM+
│   │   ├── field_extractor.py      # Solver alias → canonical field name resolution
│   │   ├── bc_encoder.py           # BC type encoding (fixed, pinned, symmetry, load, contact)
│   │   └── quality_inspector.py    # NaN/Inf, skewness, degenerate elements
│   ├── analyst_agent/
│   │   ├── analyst_agent.py        # Auto-classification → ProblemCard
│   │   ├── problem_card.py         # ProblemCard factory
│   │   └── problem_classifier.py   # Physics/mesh/material type rules
│   ├── selector_agent/
│   │   ├── deep_thinking_selector.py # Weighted model scoring + RAG history injection
│   │   ├── model_registry.py        # Loads configs/model_registry.yaml
│   │   ├── scoring_engine.py        # Per-criterion score computation
│   │   ├── discovery_agent.py       # GitHub / arXiv new architecture scan
│   │   └── github_scanner.py        # GitHub API model discovery
│   ├── trainer_agent/
│   │   ├── trainer_agent.py         # Dispatches training; MoE support
│   │   ├── training_loop.py         # Epoch loop, early stopping, checkpointing
│   │   ├── physics_loss.py          # 5-term physics penalty loss
│   │   └── auto_configurator.py     # Batch size, LR, scheduler selection
│   ├── evaluator_agent/
│   │   ├── evaluator_agent.py       # R², Rel L2, max point error, inference time
│   │   ├── metrics.py               # Metric computation functions
│   │   └── failure_diagnosis.py     # Which metric failed and by how much
│   ├── physics_agent/
│   │   ├── physics_master.py        # ThreadPoolExecutor, 5 agents in parallel
│   │   ├── equilibrium_agent.py     # ‖F_int − F_ext‖ / ‖F_ext‖
│   │   ├── stress_strain_agent.py   # σ=C:ε, von Mises yield, tensor symmetry
│   │   ├── compatibility_agent.py   # ε = sym(∇u) via cKDTree finite differences
│   │   ├── boundary_condition_agent.py # Fixed, applied load, symmetry, contact
│   │   └── material_agent.py        # Hooke / Neo-Hookean / J2 / damage
│   ├── iteration_agent/
│   │   ├── iteration_agent.py       # LLM + RAG → fix strategy selection
│   │   └── fix_strategies.py        # Lambda multipliers, BC re-encoding, model skip
│   ├── model_architect/
│   │   ├── architect_agent.py       # Failure analysis → template / novel design
│   │   ├── architecture_dna.py      # ArchitectureDNA, BlockType enum, from_llm_json()
│   │   ├── physics_block_library.py # PyTorch blocks: MambaBlock, SpectralNormBlock, etc.
│   │   ├── nas_engine.py            # Optuna NAS: block types + hyperparameters
│   │   ├── code_generator.py        # LLM → PyTorch code + validation pipeline
│   │   └── model_validator.py       # AST + import + forward + NaN + gradient check
│   ├── rag/
│   │   ├── rag_retriever.py         # Central retriever; 5 query methods
│   │   ├── vector_store.py          # FAISS IndexFlatIP; 5 named indices; persistence
│   │   ├── embedding_service.py     # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   │   └── document_builder.py      # Object → indexable text string converters
│   ├── self_learning/
│   │   ├── self_learning_updater.py # Post-run indexing into SQLite + RAG
│   │   ├── knowledge_base.py        # Lambda recommendation (RAG → SQL fallback)
│   │   ├── pattern_recognizer.py    # Geometry pattern matching via vector search
│   │   ├── performance_tracker.py   # Per-model success rate, failure streaks
│   │   └── experience_memory.py     # High-level memory access wrapper
│   ├── verifier_agent/
│   │   └── verifier_agent.py        # Claude Opus 4.7 physics certificate
│   ├── saver_agent/
│   │   └── saver_agent.py           # Model + metadata persistence
│   ├── dataset_agent/
│   │   ├── dataset_orchestrator.py  # --search-datasets flow controller
│   │   ├── dataset_searcher.py      # HuggingFace + Zenodo + curated list
│   │   ├── dataset_downloader.py    # Download + extract
│   │   └── dataset_validator.py     # FEA field presence check
│   └── shared/
│       └── llm_factory.py           # get_dev_llm() / get_verifier_llm()
│
├── models/
│   ├── base_model.py                # BaseSurrogateModel interface
│   └── architectures/               # 13 model files (see table above)
│
├── memory/
│   ├── run_database.py              # SQLite CRUD + RAG indexing hooks
│   └── model_genealogy.py           # Custom model lineage tracking
│
├── configs/
│   ├── base_config.yaml             # System, LLM, thresholds, training, physics lambdas
│   ├── model_registry.yaml          # 12 reference model entries with scores
│   ├── rag_config.yaml              # RAG enabled, indices, similarity threshold
│   └── architect_config.yaml        # Architect agent template definitions
│
├── scripts/
│   ├── generate_synthetic_fea_data.py  # Kirsch plate-with-hole generator (200 cases)
│   └── test_synthetic_pipeline.py      # End-to-end smoke test with SCF normalization
│
└── tests/                           # 13 test files, 100+ individual tests
```

---

## Key Invariants

These constraints are enforced throughout the codebase — violations cause runtime errors:

1. **FEA-only physics fields**: `PhysicsReport` has `equilibrium_passed`, `stress_strain_passed`, `compatibility_passed`, `boundary_conditions_passed`, `material_passed`. There are no CFD fields.

2. **`ProblemCard` has no `re_number` or `turbulence_model`**. Use `material_model` and `loading_type`.

3. **All loaders must return a unified schema dict (single case) or a list of dicts (multi-case)**. `DataAgent._load_data` uses `results.extend()` for list returns and `results.append()` for dict returns.

4. **`RAGRetriever` is always optional**: every agent checks `if self.retriever and self.retriever.ready` before calling RAG methods. The system runs in SQL-only mode if FAISS/sentence-transformers are not installed.

5. **Config loading always merges two files**: `base_config.yaml` first, then `rag_config.yaml` on top. Never read config from only one file.

6. **Dev and Verifier LLMs are never swapped**: code generation and physics certificates always use Claude Opus 4.7 (`get_verifier_llm()`). Analysis and ranking use `get_dev_llm()`.
