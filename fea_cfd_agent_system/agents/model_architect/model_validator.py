"""Model validator — validates generated model code compiles and runs."""

import ast
import torch
import tempfile
import importlib.util
from loguru import logger


def validate_model_code(code: str, input_dim: int = 3,
                          output_dim: int = 4) -> tuple:
    """
    Validate generated model code.
    Returns (is_valid: bool, error_message: str).
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        spec   = importlib.util.spec_from_file_location("val_model", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"Import error: {e}"

    model_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and issubclass(obj, torch.nn.Module)
                and obj is not torch.nn.Module):
            model_class = obj
            break

    if not model_class:
        return False, "No nn.Module subclass found"

    try:
        model = model_class(input_dim=input_dim, output_dim=output_dim)
        dummy = torch.randn(1, 50, input_dim)
        with torch.no_grad():
            out = model(dummy)
        if out.shape[-1] != output_dim:
            return False, f"Output dim mismatch: {out.shape[-1]} != {output_dim}"
    except Exception as e:
        return False, f"Runtime error: {e}"

    return True, "OK"
