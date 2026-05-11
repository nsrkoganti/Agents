# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, from fea_cfd_agent_system/)
pip install -e fea_cfd_agent_system/

# Run the full pipeline
cd fea_cfd_agent_system
python main.py --data path/to/result.rst --solver ansys --physics FEA_static_linear
python main.py --data result.odb --solver abaqus --physics FEA_static_nonlinear
python main.py --data result.frd --solver calculix --physics FEA_dynamic
python main.py --search-datasets --physics FEA_static_linear   # dataset discovery mode
python main.py --data data.npz --solver auto --physics FEA_static_linear  # NPZ batch

# Generate and test with synthetic data
cd fea_cfd_agent_system
python scripts/generate_synthetic_fea_data.py --cases 200 --out data/synthetic_fea/plate_with_hole.npz
python scripts/test_synthetic_pipeline.py --cases 200 --epochs 300 --target von_mises

# Run tests
cd fea_cfd_agent_system
pytest tests/ -v                          # all tests
pytest tests/test_rag.py -v               # RAG layer
pytest tests/test_fea_physics_agents.py   # physics checks
pytest tests/test_moe_surrogate.py        # MoE model
pytest tests/test_novel_architect.py      # architect + NAS
pytest tests/test_fea_models.py           # all architecture forward passes

# Environment variables (required for full LLM operation)
export ANTHROPIC_API_KEY=...   # always required (verifier LLM + dev fallback)
export GEMINI_API_KEY=...      # optional; preferred for dev LLM (free 15 RPM / 1500 RPD)
```

## Architecture

### Agent Pipeline (LangGraph StateGraph)

`main.py` → `MasterOrchestrator` builds and runs a stateful graph. All agents read/write `AgentSystemState` (a single dataclass in `agents/orchestrator/agent_state.py`).

```
DataAgent → [DatasetOrchestrator] → AnalystAgent → SelectorAgent → TrainerAgent
              (if no data path)                                          │
                                                                  EvaluatorAgent
                                                                  ┌──────┴──────┐
                                                            PhysicsMasterAgent   IterationAgent
                                                                  │              ┌────┴──────────┐
                                                            VerifierAgent    retry/select_new/architect
                                                                  │
                                                            SaverAgent → SelfLearningUpdater
```

Routing is conditional: `EvaluatorAgent` routes to Physics (pass) or Iteration (fail); `PhysicsMasterAgent` routes to Verifier (pass) or Iteration (fail); `IterationAgent` returns to Trainer, SelectNew, or triggers the Architect. Max 24 total attempts.

### Key State Objects (`agents/orchestrator/agent_state.py`)

- **`AgentSystemState`** — the single object passed through every node. Contains all intermediate outputs.
- **`ProblemCard`** — analyst output: `physics_type` (PhysicsType enum), `mesh_type`, `material_model`, `loading_type`, `element_type`, `solver_source`, `accuracy_threshold`.
- **`PhysicsReport`** — physics master output: 5 FEA checks (`equilibrium_passed`, `stress_strain_passed`, `compatibility_passed`, `boundary_conditions_passed`, `material_passed`) + `overall_passed` + `physics_lambda_updates`.
- **`ModelCandidate`** — scored candidate from model registry.
- **`IterationRecord`** — one attempt: model, result, failure reason, fix applied.

### LLM Factory (`agents/shared/llm_factory.py`)

Two LLM roles, never mix them:
- **Dev LLM** (`get_dev_llm()`): Gemini 2.5 Flash → Claude Sonnet 4.6 fallback. Used for analysis, ranking, iteration diagnosis, and dataset discovery.
- **Verifier LLM** (`get_verifier_llm()`): Always Claude Opus 4.7. Used for code generation and physics certificate validation.

### Physics Agents (`agents/physics_agent/`)

Five FEA-specific checks run in parallel via `ThreadPoolExecutor` inside `PhysicsMasterAgent`:

| Agent | Check | Threshold |
|---|---|---|
| `EquilibriumAgent` | `‖F_int − F_ext‖ / ‖F_ext‖` | 1e-5 (linear), 1e-4 (nonlinear) |
| `StressStrainAgent` | `‖σ − C:ε‖`, von Mises ≤ yield, tensor symmetry | 2% |
| `CompatibilityAgent` | `ε = sym(∇u)` via finite differences (cKDTree) | 1e-7 |
| `BoundaryConditionAgent` | Fixed supports `‖u‖<1e-6`, reaction balance | 1e-6 |
| `MaterialAgent` | Hooke / Neo-Hookean / J2 elastoplastic | model-specific |

When a check fails, `physics_lambda_updates` multiplies the corresponding penalty weight (see `lambda_multipliers` in `configs/base_config.yaml`). The IterationAgent re-trains with the updated lambdas.

### ML Models (`models/architectures/`)

All inherit `BaseSurrogateModel` with interface `(B, N, in_dim) → (B, N, out_dim)`:

| File | Model | Best for |
|---|---|---|
| `transolver_3.py` | Transolver-3 | Industrial FEA, 1M+ nodes |
| `transolver_pp.py` | Transolver++ | General FEA, eidetic states |
| `eagnn.py` | EAGNN | Unstructured tet, nonlinear stress |
| `mgn_transformer.py` | MeshGraphNet-Transformer | Plasticity, large deformation |
| `gnss.py` | GNSS | Structural dynamics, local frames |
| `deeponet.py` | GS-PI-DeepONet | Small data (<100 samples) |
| `fno_surrogate.py` | Factorized FNO | Structured grids |
| `gnn_surrogate.py` | MeshGraphNet | General unstructured baseline |
| `pinn.py` | PINN | Physics-constrained, small data |
| `moe_surrogate.py` | MixtureOfExperts | Spatially heterogeneous fields |

Model selection is scored by `SelectorAgent` using weights from `configs/base_config.yaml` (`selector.score_weights`) and the model registry `configs/model_registry.yaml`.

### Custom Architect (`agents/model_architect/`)

Triggered after 12 failed attempts. Flow:
1. `ArchitectAgent._analyze_failures()` → LLM diagnoses why all models failed
2. `_choose_template()` picks from 5 DNA templates (pinn, transolver, gnn, operator, hybrid)
3. After all 5 templates exhausted: `_design_novel_architecture()` → LLM designs a novel block sequence as JSON → `ArchitectureDNA.from_llm_json()`
4. `NASEngine.refine_dna()` (Optuna) searches block types + hyperparameters
5. `CodeGenerator.generate()` → LLM writes complete PyTorch class
6. `CodeGenerator.validate()` → AST parse + import + forward pass + NaN check + gradient flow check
7. Saved to SQLite `custom_models` table for future reuse

**`ArchitectureDNA`** — JSON-serializable block sequence. `BlockType` enum lists all valid blocks including `MAMBA_BLOCK`, `CONV_NEXT_BLOCK`, `CROSS_ATTENTION`, `SPECTRAL_NORM`. Implementations in `agents/model_architect/physics_block_library.py`.

### RAG Layer (`agents/rag/`)

Built on FAISS `IndexFlatIP` (cosine via L2 normalisation) + `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB, downloaded on first run). Five named indices, each persisted to `memory/rag_indices/{name}.faiss` + `.meta.pkl`:

| Index | What is indexed | Used by |
|---|---|---|
| `geometry_index` | problem description + physics/mesh type | KnowledgeBase, PatternRecognizer |
| `failure_index` | failure reason + fix tried + model name | IterationAgent |
| `physics_model_index` | model performance on specific physics | SelectorAgent |
| `lambda_index` | lambda configs that resolved physics violations | KnowledgeBase |
| `custom_dna_index` | custom model DNA block sequences | ArchitectAgent, CodeGenerator |

**`RAGRetriever`** is created once in `MasterOrchestrator.__init__()` and injected into all agents that need it. It gracefully no-ops if `sentence-transformers` or `faiss-cpu` are not installed.

Every `RunDatabase.save_run()`, `save_failure()`, and `save_custom_model()` also calls the retriever's indexing methods.

### Self-Learning (`agents/self_learning/`)

- **`KnowledgeBase`** — `recommend_lambda_weights()` queries RAG first, falls back to SQL; `has_seen_similar_problem()` uses cosine similarity ≥ 0.70.
- **`PatternRecognizer`** — replaces SQL `LIKE` geometry queries with `find_similar_problems()`.
- **`SelfLearningUpdater`** — runs after every completed pipeline; updates performance stats, increments model failure counts, logs the full run to SQLite.
- **`RunDatabase`** — SQLite at `memory/experience.db`. Tables: `runs`, `failures`, `model_performance`, `lambda_history`, `custom_models`, `patterns`.

### Data Loading (`agents/data_agent/`)

`DataAgent._get_loader(ext, software)` dispatches:
- `.rst/.rth` → `ANSYSLoader` (ansys-mapdl-reader)
- `.odb/.inp` → `AbaqusLoader` (abapy / meshio)
- `.frd/.dat` → `CalculiXLoader` (ccx2paraview / meshio)
- `.csv` + STAR-CCM+ → `StarCCMFEALoader`
- `.npy/.npz` → `NumpyLoader` (handles flat, stacked batch `(C,N,D)`, and object-array layouts)
- everything else → `VTKLoader`

All loaders return the same **unified schema**: `{nodes (N,3), elements (E,K), fields {displacement,stress,strain,von_mises,...}, n_nodes, n_elements, physics_type, solver_source, mesh_type, boundary_info, material_properties, load_steps}`.

`FieldExtractor` resolves solver-specific field names (e.g. `SEQV`, `vonMises`, `VM_STRESS` → `von_mises`) and computes von Mises from Voigt stress if missing.

### Config Loading (`main.py`)

`load_config()` reads `configs/base_config.yaml` then merges `configs/rag_config.yaml` on top. The merged dict is passed to all agents and `MasterOrchestrator`. Physics-type-specific thresholds live under `configs/base_config.yaml → physics.fea.*`.

### Critical Invariants

- **`PhysicsReport` fields are FEA-only**: `equilibrium_passed`, `stress_strain_passed`, `compatibility_passed`, `boundary_conditions_passed`, `material_passed`. Never reference old CFD fields (`governing_equations_passed`, `conservation_passed`, `turbulence_passed`).
- **`ProblemCard` has no `re_number` or `turbulence_model`**. Use `material_model` and `loading_type` instead.
- **All loaders must return a unified schema dict** (single case) or a list of such dicts (multi-case). `DataAgent._load_data` uses `results.extend()` for list returns (NumpyLoader) and `results.append()` for dict returns.
- **`RAGRetriever` is always optional**: every agent checks `if self.retriever and self.retriever.ready` before calling RAG methods.
- **Dev branch**: `claude/new-session-p0DjH` — all development goes here.
