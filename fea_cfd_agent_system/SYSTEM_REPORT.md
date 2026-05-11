# FEA Surrogate ML Agent System — Complete Technical Report

---

## What Is This System?

This is an **autonomous AI agent system** that takes your FEA (Finite Element Analysis) results
as input and automatically:

1. Reads and understands your simulation data
2. Selects the best ML model for the problem
3. Trains and evaluates the model
4. Validates it against real FEA physics laws
5. If it fails, automatically tries to fix or redesign
6. If everything else fails, **invents a brand-new model architecture tailored to your problem**

It runs as a pipeline — you point it at a file and it handles everything.

---

## How to Run It

```bash
# With an ANSYS result file
python main.py --data path/to/result.rst --solver ansys --physics FEA_static_linear

# With a CalculiX result
python main.py --data result.frd --solver calculix --physics FEA_dynamic

# No data? Let it find and download an FEA dataset for you
python main.py --search-datasets --physics FEA_static_linear

# Full options
python main.py --data result.vtu --solver vtk --max-attempts 12 --output-dir results/
```

**Supported FEA solvers:** ANSYS (`.rst`), Abaqus (`.odb`), CalculiX (`.frd`), STAR-CCM+ (CSV/VTK export), generic VTK/HDF5

---

## Step-by-Step Pipeline Flow

```
Your FEA File
     │
     ▼
Step 1: DATA AGENT         — read & normalize your FEA data
     │
     ▼
Step 2: ANALYST AGENT      — understand the problem (what physics? what mesh?)
     │
     ▼
Step 3: SELECTOR AGENT     — rank and pick the best ML model
     │
     ▼
Step 4: TRAINER AGENT      — train the model on your data
     │
     ▼
Step 5: EVALUATOR AGENT    — measure accuracy (R², L2 error, etc.)
     │
     ├──[FAIL]─► Step 6a: ITERATION AGENT — diagnose and fix
     │                       │
     │                       ├── retry same model (new hyperparams)
     │                       ├── pick a different model
     │                       └──[after 12 fails]─► ARCHITECT AGENT (see below)
     │
     ▼ [PASS accuracy]
Step 6: PHYSICS MASTER AGENT — verify FEA physics laws (5 checks in parallel)
     │
     ├──[FAIL]─► ITERATION AGENT (adjust physics loss weights, retrain)
     │
     ▼ [PASS physics]
Step 7: VERIFIER AGENT     — final quality certificate (Claude Opus)
     │
     ▼
Step 8: SAVER AGENT        — save model, metadata, results
     │
     ▼
Step 9: SELF-LEARNING UPDATE — store what worked in the database for next time
```

**Maximum attempts:** 24 total (configurable). The system never gives up silently — it always
explains what failed and why.

---

## Agent-by-Agent Breakdown

### Step 1 — Data Agent

Reads your FEA result file and converts it to a unified schema regardless of solver:

```
nodes          (N, 3)    — xyz coordinates of every mesh node
displacement   (N, 3)    — UX, UY, UZ predicted displacement
stress         (N, 6)    — Voigt notation: σxx σyy σzz σxy σyz σxz
von_mises      (N,)      — equivalent stress
boundary_conditions      — which nodes are fixed, loaded, symmetric
material_props           — E (Young's modulus), ν (Poisson), yield stress
```

All 4 solvers produce this same format regardless of their native file format.

---

### Step 2 — Analyst Agent

Reads the unified schema and fills in a **Problem Card** — a structured description of your problem:

| Field | Example |
|---|---|
| Physics type | `FEA_static_nonlinear` |
| Mesh type | `unstructured_tetrahedral` |
| Material model | `elastoplastic` |
| Loading type | `static` |
| Element type | `tet` |
| Accuracy target | R² ≥ 0.93, L2 ≤ 5% |

**Auto-detection logic:**

| Signal in Data | Classification |
|---|---|
| Time arrays present | `loading_type = "dynamic"` |
| Temperature field present | `loading_type = "thermal"` |
| Multiple load steps | `loading_type = "cyclic"` |
| 4 nodes/element | `element_type = "tet"` |
| 8 nodes/element | `element_type = "hex"` |
| 3 nodes/element | `element_type = "shell"` |
| yield_stress in material | `material_model = "elastoplastic"` |

---

### Step 3 — Selector Agent

Uses **Claude Opus 4.7** (deep thinking) to score and rank all available ML models for your
specific Problem Card. Scoring considers:

- Does the model support your mesh type?
- Does it have a built-in physics loss?
- What is its published L2 benchmark error?
- How fast is inference?
- How many training samples do you have?

**Available models (ranked best to most experimental):**

| Model | Best For | Speed vs FEA | Physics Loss |
|---|---|---|---|
| **Transolver-3** | Industrial-scale FEA (1M+ nodes) | ~500× | No |
| **Transolver++** | Best general FEA accuracy (ICML 2025) | ~500× | No |
| **EAGNN** | Stress concentrations, unstructured tet mesh | ~1000× | No |
| **MeshGraphNet-Transformer** | Nonlinear / plastic FEA | ~500× | No |
| **GNSS** | Structural dynamics / transient | ~200× | No |
| **GS-PI-DeepONet** | Small data (<100 samples), parametric PDEs | ~100× | Yes |
| **Factorized FNO** | Structured grids only | ~100× | No |
| **MeshGraphNet** | General unstructured mesh baseline | ~1000× | No |
| **Hybrid Transolver+PINN** | Physics-constrained problems | ~300× | Yes |
| **PINN** | Very small data, strong physics constraints | ~50× | Yes |
| **MLP** | Tabular / low-DOF baseline | ~100× | No |

---

### Step 4 — Trainer Agent

Trains the selected model using PyTorch with:

- Adam optimizer, LR = 0.001
- Cosine LR schedule
- Early stopping (patience = 20 epochs)
- Max 2000 epochs
- Gradient clipping (max norm = 1.0)

**Physics loss penalty** (if model supports it):

```
total_loss = data_MSE
           + λ_eq   × equilibrium_residual
           + λ_bc   × BC_violation²
           + λ_sym  × stress_tensor_symmetry
           + λ_comp × strain_compatibility
           + λ_const × constitutive_law_error
```

The λ weights are dynamically increased (×2 to ×5) every time a physics check fails.

---

### Step 5 — Evaluator Agent

Measures three metrics against your accuracy targets:

| Metric | What It Measures | Threshold (linear) | Threshold (nonlinear) |
|---|---|---|---|
| **R² score** | Overall prediction quality | ≥ 0.95 | ≥ 0.93 |
| **Relative L2 error** | Field-level accuracy | ≤ 3% | ≤ 5% |
| **Max point error** | Worst-case single-node error | ≤ 10% | ≤ 10% |

If any metric fails → routes to **Iteration Agent**.

---

### Step 6a — Iteration Agent

Diagnoses the failure and chooses a fix strategy:

| Failure Pattern | Fix Applied |
|---|---|
| Underfitting (R² < 0.5) | Increase hidden_dim, add more layers |
| Overfitting | Add dropout, reduce model size |
| Physics violation — BC | Multiply λ_bc by 5× |
| Physics violation — equilibrium | Multiply λ_eq by 2× |
| Wrong mesh type for model | Select next model from shortlist |
| All standard models exhausted | Trigger **Architect Agent** |

---

### Step 6b — Physics Master Agent

Runs **5 FEA physics checks in parallel** on the model's predictions:

#### Check 1 — Equilibrium

```
||F_internal - F_external|| / ||F_external|| < 0.00001   (linear static)
                                              < 0.0001    (nonlinear)
```

Recovers nodal forces from predicted stress via virtual work: `F_int = ∫ B^T σ dV`

#### Check 2 — Stress-Strain (Constitutive Law)

```
||σ - C:ε|| / ||σ|| < tolerance
σ_vonMises ≤ yield_stress     (if elastoplastic)
|σ_ij - σ_ji| < 1e-8          (tensor symmetry)
```

#### Check 3 — Compatibility

```
ε = sym(∇u) via finite differences on predicted displacements
||u_nodeA - u_nodeB|| < 1e-7   (displacement continuity across shared nodes)
```

#### Check 4 — Boundary Conditions

- Fixed nodes: `||u|| < 1e-6`
- Applied loads: reaction forces balance applied forces (Newton's 3rd law)
- Symmetry planes: normal displacement ≈ 0
- Contact nodes: penetration = 0, contact pressure ≥ 0

#### Check 5 — Material Model

- Linear elastic: `σ = E·ε / (1+ν)` consistency
- Hyperelastic Neo-Hookean: `W = μ/2(I₁−3) − μln(J) + λ/2·ln²(J)`
- Elastoplastic J2: von Mises yield surface + isotropic hardening `σ_y = σ_y0 + H·ε_p`

---

### Step 7 — Verifier Agent

Final sanity check using **Claude Opus 4.7** as a physics judge. Reads the full evaluation
report + physics report and issues a binary pass/fail certificate with a written explanation.

---

### Step 8 — Saver Agent

Saves to disk:

- Trained model weights (`.pt`)
- Full metadata JSON (Problem Card, metrics, physics report, training config)
- Model architecture description
- Results logged to MLflow experiment tracker

---

### Step 9 — Self-Learning Updater

Writes this run's outcome to the **SQLite knowledge base** so future runs are smarter:

- Which model worked for which physics type / mesh type combination
- What hyperparameters performed best
- What failure patterns occurred and what fixed them
- Custom model architectures that succeeded (automatically reused next time)

---

## Feature A — Mixture of Experts (MoE)

### Why It Exists

FEA fields are spatially heterogeneous. A notch tip has stress concentrations that need a
GNN's local mesh awareness. The smooth bulk interior is well-served by Transolver/FNO. Boundary
nodes need physics-constrained predictions from a PINN. **No single model is optimal everywhere.**

MoE solves this by running 3 expert models simultaneously and letting a small neural network
decide, **at every mesh node individually**, which expert to trust:

```
output[node] = gate_weight_1[node] × Expert1(node_features)
             + gate_weight_2[node] × Expert2(node_features)
             + gate_weight_3[node] × Expert3(node_features)
```

The gate weights always sum to 1.0 per node (softmax). Fully differentiable — trains end-to-end.

### How It's Built

**`NodeGatingNetwork`** — a 2-layer MLP:

```
(B, N, in_dim) → Linear → GELU → Linear → GELU → Linear → softmax → (B, N, 3)
```

Takes the same node features as the experts, outputs one probability per expert per node.

**`MixtureOfExpertsSurrogate` forward pass:**

1. Runs all 3 experts → each produces `(B, N, out_dim)`
2. Stacks outputs → `(B, N, 3, out_dim)`
3. Multiplies by gate weights → weighted sum → `(B, N, out_dim)`
4. Result: each node independently blends expert outputs

### Training — 3 Phases

| Phase | What Trains | Why |
|---|---|---|
| **Phase 1** | Each expert independently | Ensure every expert is competent before gating |
| **Phase 2** | Only the gating network (experts frozen) | Stable: experts don't shift while gating learns |
| **Phase 3** | Everything end-to-end at LR = 1e-4 | Fine-tune the joint system together |

### Physics Loss in MoE

Each expert's physics violations are weighted by how much that expert is being used:

```
total_physics_loss = Σ_k (mean_gate_k × expert_k_physics_loss)
                   + 0.01 × variance(mean_gate)    ← load-balance penalty
```

The **load-balance penalty** (weight = 0.01) prevents one expert from always winning and the
others being ignored (called "expert collapse").

### Visualising Expert Attribution

```python
attr = moe.get_expert_attribution(x)   # returns (B, N, 3)
```

Returns a heatmap showing, for every mesh node, how much each expert contributed. Plot on your
mesh to see:

- **Blue regions** → Expert 1 dominated (e.g. smooth FNO / Transolver regions)
- **Green regions** → Expert 2 dominated (e.g. GNN at stress concentrations)
- **Red regions** → Expert 3 dominated (e.g. PINN at constrained boundary nodes)

---

## Feature B — Novel Architecture Generation

### The Problem This Solves

The system has 5 standard model templates. Sometimes a unique combination of physics type, mesh
topology, material model, or data size means none of the 5 work well enough. In that case,
instead of giving up, the system **designs a brand-new architecture from scratch**.

This is how the system can invent its own Transolver variant tailored specifically to your problem.

### The Full Novel Architecture Flow

```
12+ failed attempts with standard models
           │
           ▼
PHASE 1: Try 5 standard templates (existing behavior)
  pinn_dna → transolver_dna → fno_dna → gnn_dna → hybrid_dna
  (with NAS hyperparameter tuning via Optuna)
           │
           │ (if ALL 5 fail)
           ▼
PHASE 2: LLM designs a novel block sequence
  Claude Opus 4.7 reads:
    - Last 8 failure records (model, R², failure reason, fix tried)
    - Problem Card (physics type, mesh type, data size, output targets)
    - ALL available block types (see table below)
  Returns a JSON block sequence describing a new architecture
           │
           ▼
PHASE 3: NAS (Optuna, 30 trials) refines the design
  Searches: block types, hidden_dim, n_layers, n_slices, n_heads,
            dropout, residual connections, normalization type
           │
           ▼
PHASE 4: Code Generator produces a complete PyTorch nn.Module
  Validated: AST parse + import check + runtime shape check
             + NaN / Inf detection + gradient flow check
           │
           ▼
PHASE 5: Train → Evaluate → Physics check (same pipeline as always)
           │
           ▼
PHASE 6: Save to knowledge base
  SQLite custom_models table stores:
    - DNA (complete block sequence as JSON)
    - Full Python nn.Module code
    - R² score (updated after training)
    - Physics type and generation number
  Future runs with similar problems reuse this model automatically
```

### Available Building Blocks for Novel Architectures

The LLM can combine **any** of these blocks in any sequence:

| Block | What It Does | Best For |
|---|---|---|
| `coord_embed` | Fourier coordinate embedding — encodes (x,y,z) into high-dimensional features | All mesh models — always include first |
| `bc_encoder` | Encodes boundary condition type + value | Models that need to respect BCs explicitly |
| `physics_attention` | Transolver-style: groups N nodes into S=32 "physics slices", attention is O(S²) not O(N²) | Large meshes — the key Transolver innovation |
| `mamba_block` | State-space model scan — O(N) complexity, linear in number of nodes | Very large meshes, long-range dependencies |
| `conv_next_block` | Depthwise Conv1d + pointwise + LayerNorm | Fast local feature extraction |
| `cross_attention` | Cross-attention from learnable physics query tokens to mesh nodes | Querying global physics state at specific locations |
| `fourier_layer` | Spectral domain multiplication | Structured grids only |
| `graph_conv` | GNN message passing using mesh connectivity | Unstructured meshes, local topology |
| `graph_attention` | Attention-weighted GNN message passing | Unstructured meshes with variable importance |
| `residual_connection` | Skip connection — adds input back to output | Deep networks — prevents gradient vanishing |

### Example: What the LLM Might Design

For a failing nonlinear FEA problem on a large tet mesh with stress concentrations, Claude
Opus might return:

```json
{
  "name": "CustomMambaTransolver",
  "family": "hybrid",
  "designed_for": "FEA_static_nonlinear",
  "has_physics_loss": true,
  "physics_loss_types": ["equilibrium", "bc"],
  "input_blocks": [
    {"type": "coord_embed",  "hidden_dim": 256},
    {"type": "bc_encoder",   "hidden_dim": 256}
  ],
  "core_blocks": [
    {"type": "mamba_block",       "hidden_dim": 256},
    {"type": "physics_attention", "hidden_dim": 256, "n_slices": 32},
    {"type": "mamba_block",       "hidden_dim": 256},
    {"type": "cross_attention",   "hidden_dim": 256, "n_queries": 64}
  ],
  "output_blocks": [
    {"type": "linear", "hidden_dim": 128},
    {"type": "gelu"},
    {"type": "linear", "hidden_dim": 6}
  ]
}
```

This is a completely new architecture — not Transolver, not PINN, not FNO. It combines Mamba's
linear-complexity scan with Transolver's physics-aware attention, designed specifically for
your failure pattern.

### Re-use Across Sessions

Once a novel architecture succeeds, it is saved to SQLite:

```
custom_models table:
  model_id     → "custom_CustomMambaTransolver_a1b2c3d4"
  name         → "CustomMambaTransolver"
  dna_json     → full block sequence as JSON (reloadable)
  code         → complete Python nn.Module code
  physics_type → "FEA_static_nonlinear"
  r2_score     → 0.971   (updated after training)
  generation   → 2       (generation 1 = standard templates)
```

Next time you run the system with a similar problem, it detects the match and starts with your
proven custom architecture instead of starting over from the standard templates.

---

## LLM Usage Summary

| Task | Model | Why |
|---|---|---|
| Failure analysis | Gemini 2.5 Flash (free) | Fast, called frequently during iteration |
| Model selection / deep thinking | Claude Opus 4.7 | Needs strong reasoning about trade-offs |
| Novel architecture design | Claude Opus 4.7 | Needs creativity + deep ML knowledge |
| PyTorch code generation | Claude Opus 4.7 | Must write syntactically and physically correct code |
| Physics verification certificate | Claude Opus 4.7 | Final quality gate |
| Fallback (no Gemini key) | Claude Sonnet 4.6 | Always available |

Set `GEMINI_API_KEY` in your environment for free fast inference on routine analysis steps.
`ANTHROPIC_API_KEY` is always required for the verifier and architect roles.

---

## Key Files Reference

```
fea_cfd_agent_system/
├── main.py                                     ← Entry point / CLI
│
├── agents/
│   ├── orchestrator/
│   │   ├── master_orchestrator.py              ← LangGraph pipeline builder & router
│   │   └── agent_state.py                      ← All shared state (ProblemCard, PhysicsReport, etc.)
│   │
│   ├── data_agent/
│   │   ├── ansys_loader.py                     ← ANSYS .rst reader
│   │   ├── abaqus_loader.py                    ← Abaqus .odb reader
│   │   ├── calculix_loader.py                  ← CalculiX .frd reader
│   │   └── starccm_fea_loader.py               ← STAR-CCM+ CSV/VTK reader
│   │
│   ├── selector_agent/
│   │   └── deep_thinking_selector.py           ← Ranks ML models using Claude Opus
│   │
│   ├── physics_agent/
│   │   ├── equilibrium_agent.py                ← F_int vs F_ext check
│   │   ├── stress_strain_agent.py              ← σ = C:ε, von Mises, tensor symmetry
│   │   ├── compatibility_agent.py              ← ε = sym(∇u) check
│   │   ├── boundary_condition_agent.py         ← BC enforcement check
│   │   ├── material_agent.py                   ← Constitutive law check
│   │   └── physics_master.py                   ← Runs all 5 checks in parallel
│   │
│   ├── model_architect/
│   │   ├── architect_agent.py                  ← Designs new architectures when all else fails
│   │   ├── architecture_dna.py                 ← BlockType enum + ArchitectureDNA schema
│   │   ├── physics_block_library.py            ← PyTorch implementations of all block types
│   │   ├── code_generator.py                   ← LLM → validated PyTorch nn.Module code
│   │   └── nas_engine.py                       ← Optuna hyperparameter + block-type search
│   │
│   └── self_learning/
│       ├── knowledge_base.py                   ← "Have we seen this problem before?"
│       └── self_learning_updater.py            ← Writes run outcomes to SQLite DB
│
├── models/architectures/
│   ├── moe_surrogate.py                        ← Mixture of Experts (NEW Feature A)
│   ├── transolver_3.py                         ← Transolver-3 (arxiv:2602.04940, 2025)
│   ├── transolver_pp.py                        ← Transolver++ (ICML 2025)
│   ├── eagnn.py                                ← Edge-Attributed GNN
│   ├── mgn_transformer.py                      ← MeshGraphNet-Transformer
│   ├── gnss.py                                 ← Graph Network Structural Simulator
│   ├── deeponet.py                             ← GS-PI-DeepONet
│   ├── fno_surrogate.py                        ← Factorized FNO
│   ├── gnn_surrogate.py                        ← MeshGraphNet baseline
│   ├── pinn.py                                 ← Physics-Informed Neural Network
│   ├── hybrid_model.py                         ← Transolver + PINN hybrid
│   └── mlp_surrogate.py                        ← MLP baseline
│
├── memory/
│   └── run_database.py                         ← SQLite: run history + custom models
│
├── configs/
│   ├── base_config.yaml                        ← Main config (thresholds, LLMs, training)
│   └── model_registry.yaml                     ← All model metadata + MoE entry
│
└── tests/
    ├── test_moe_surrogate.py                   ← MoE tests (NEW Feature A)
    ├── test_novel_architect.py                 ← Novel architecture tests (NEW Feature B)
    ├── test_fea_physics_agents.py              ← Physics check tests
    ├── test_fea_data_loaders.py                ← Data loader tests
    └── test_fea_models.py                      ← Model architecture tests
```

---

## Summary of What Was Built

| Component | Status | What It Does |
|---|---|---|
| `moe_surrogate.py` | **NEW** | Per-node soft-weighted combination of 3 expert models |
| `architect_agent.py` — novel design method | **NEW** | LLM designs architectures beyond the 5 fixed templates |
| `architecture_dna.py` — new block types | **NEW** | Mamba, ConvNeXt, CrossAttention blocks added |
| `nas_engine.py` — block-type search | **NEW** | NAS now searches block types, not just hyperparameters |
| `code_generator.py` — enhanced validation | **NEW** | NaN detection + gradient flow checks added |
| All 5 physics agents | **EXISTING** | Equilibrium, stress-strain, compatibility, BC, material |
| All 11 ML model architectures | **EXISTING** | Transolver-3, EAGNN, MeshGraphNet-T, GNSS, DeepONet, etc. |
| Self-learning SQLite database | **EXISTING** | Knowledge accumulates across all runs automatically |
| LangGraph pipeline | **EXISTING** | Orchestrates all agents with conditional routing |
