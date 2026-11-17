"""Loads the student-provided custom implementation.

The module is chosen by the GO2_CUSTOM_IMPL environment variable: a file name
inside go2_deploy/custom_implementation/ (default: go2trot_base.py).
"""

import importlib.util
import os
from pathlib import Path

DEFAULT_CUSTOM_IMPL = "go2trot_base.py"


def get_custom_impl_module(name=None):
    name = name or os.environ.get("GO2_CUSTOM_IMPL", DEFAULT_CUSTOM_IMPL)
    repo_path = Path(__file__).resolve().parent.parent
    path = Path(repo_path / "custom_implementation" / name)
    if not path.is_file():
        raise ImportError(f"No custom implementation at {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CUSTOM_IMPL_NAME = name
    return module


custom_impl = get_custom_impl_module()
