"""
Code Generator — takes an ArchitectureDNA and generates
a complete, working Python file (nn.Module subclass).
"""

import json
import ast
import torch
import tempfile
import importlib.util
from pathlib import Path
from loguru import logger
from agents.shared.llm_factory import get_verifier_llm
from agents.model_architect.architecture_dna import ArchitectureDNA


class CodeGenerator:
    """
    Converts ArchitectureDNA into working PyTorch model code.
    Uses LLM to write the forward() pass and any custom logic.
    Validates generated code actually imports and runs.
    """

    def __init__(self):
        self.llm = get_verifier_llm(max_tokens=4000)

    def generate(self, dna: ArchitectureDNA,
                  problem_card_dict: dict) -> str:
        """Generate complete Python code for a model from its DNA."""
        prompt = self._build_prompt(dna, problem_card_dict)
        response = self.llm.invoke(prompt)
        code = self._extract_code(response.content)
        logger.info(f"Generated code for {dna.name} ({len(code)} chars)")
        return code

    def _build_prompt(self, dna: ArchitectureDNA,
                       problem_card: dict) -> str:
        block_descriptions = self._describe_blocks(dna)
        physics_loss_desc  = self._describe_physics_loss(dna)

        return f"""
You are an expert PyTorch neural network architect specializing in
physics simulation surrogate models.

Write a complete, working PyTorch model class for this architecture.

ARCHITECTURE NAME: {dna.name}
FAMILY: {dna.family}
DESIGNED FOR: {dna.designed_for}
MESH TYPE SUPPORTED: {dna.mesh_type}
PARENT ARCHITECTURES: {dna.parent_names}

PROBLEM CONTEXT:
{json.dumps(problem_card, indent=2)}

ARCHITECTURE BLOCKS:
{block_descriptions}

PHYSICS LOSS:
{physics_loss_desc}

REQUIREMENTS:
1. Class must be named {dna.name.replace('-','_').replace(' ','_')}
2. Must inherit from torch.nn.Module
3. __init__ must accept: input_dim, output_dim, and all architecture params
4. forward() must accept: x (tensor of shape (batch, n_nodes, input_dim))
5. forward() must return: tensor of shape (batch, n_nodes, output_dim)
6. Must use the EXACT physics block implementations from agents.model_architect.physics_block_library:
   - PhysicsAttentionBlock for physics_attention blocks
   - FourierLayer for fourier_layer blocks
   - GraphConvBlock for graph_conv blocks
   - CoordinateEmbedding for coord_embed blocks
   - BoundaryConditionEncoder for bc_encoder blocks
7. Import them at top: from agents.model_architect.physics_block_library import ...
8. If has_physics_loss=True, implement compute_physics_loss() method that
   returns a dict of loss terms: {{"continuity": tensor, "bc": tensor, ...}}
9. Add docstring explaining the architecture
10. Code must be complete and runnable — no placeholder comments

{"SPECIAL FOR TRANSOLVER FAMILY: Use PhysicsAttentionBlock exactly as designed." if "transolver" in dna.family.lower() or "transformer" in dna.family.lower() else ""}
{"SPECIAL FOR PINN FAMILY: Use tanh activations, coordinate embedding, and physics loss." if dna.family == "pinn" else ""}
{"SPECIAL FOR HYBRID: Combine architectures intelligently." if dna.family == "hybrid" else ""}

Output ONLY the Python code. No markdown. No explanation. Just the class.
Start with imports.
"""

    def _describe_blocks(self, dna: ArchitectureDNA) -> str:
        lines = []
        for section, blocks in [
            ("Input Processing", dna.input_processing),
            ("Core Blocks",      dna.core_blocks),
            ("Output",           dna.output_processing),
        ]:
            lines.append(f"\n{section}:")
            for b in blocks:
                lines.append(
                    f"  - {b.block_type.value}: "
                    f"hidden_dim={b.hidden_dim}, "
                    f"n_slices={b.n_slices}, "
                    f"n_heads={b.n_heads}, "
                    f"n_modes={b.n_modes}, "
                    f"residual={b.residual}"
                )
        return "\n".join(lines)

    def _describe_physics_loss(self, dna: ArchitectureDNA) -> str:
        if not dna.has_physics_loss:
            return "None — purely data-driven model"
        return (
            f"Types: {dna.physics_loss_types}\n"
            "Implement compute_physics_loss(pred, coords) returning dict of tensors.\n"
            "Continuity: MSE(div(velocity)) — must be ~0 for incompressible CFD\n"
            "BC: MSE(velocity at wall nodes) — must be 0 for no-slip\n"
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        return code.strip()

    def validate(self, code: str, input_dim: int = 3,
                  output_dim: int = 4) -> tuple:
        """
        Validate generated code:
        1. Parses without SyntaxError
        2. Can be imported
        3. Can be instantiated
        4. forward() runs on dummy input
        Returns (is_valid: bool, error: str)
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            spec   = importlib.util.spec_from_file_location("generated_model", tmp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return False, f"Import error: {e}"

        model_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                    issubclass(attr, torch.nn.Module) and
                    attr is not torch.nn.Module):
                model_class = attr
                break

        if model_class is None:
            return False, "No nn.Module subclass found in generated code"

        try:
            model = model_class(input_dim=input_dim, output_dim=output_dim)
        except Exception as e:
            return False, f"Instantiation error: {e}"

        try:
            dummy = torch.randn(2, 100, input_dim)
            with torch.no_grad():
                out = model(dummy)
            if out.shape[-1] != output_dim:
                return False, f"Wrong output dim: got {out.shape[-1]}, expected {output_dim}"
        except Exception as e:
            return False, f"Forward pass error: {e}"

        # NaN / Inf check on output
        if not torch.isfinite(out).all():
            return False, "Output contains NaN or Inf values"

        # Gradient flow check — ensure at least one parameter receives gradients
        try:
            model.train()
            dummy_train = torch.randn(2, 100, input_dim)
            loss = model(dummy_train).mean()
            loss.backward()
            has_grad = any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad
            )
            if not has_grad:
                return False, "No parameter received a finite gradient (dead network)"
        except Exception as e:
            return False, f"Gradient flow check failed: {e}"

        return True, "OK"
