"""Small import shim for the parts of diffusers used by robomimic DP.

The old diffusers==0.11.1 package eagerly imports pipelines from
``diffusers.__init__``. That path imports ``transformers`` even when we only
need the DDPM/DDIM schedulers and EMA helper. In this workspace the
transformers import is fragile, so importing the tiny scheduler modules through
the public package can randomly kill long training jobs.

This module creates a lightweight ``diffusers`` package namespace that points
at the installed diffusers files, then loads only the scheduler and training
utility modules robomimic needs. It deliberately avoids executing
``diffusers/__init__.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from importlib.machinery import PathFinder
from pathlib import Path


def _diffusers_root() -> Path:
    spec = PathFinder.find_spec("diffusers")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not locate installed diffusers package")
    return Path(next(iter(spec.submodule_search_locations)))


def _ensure_lightweight_package(root: Path) -> None:
    package = sys.modules.get("diffusers")
    if package is None:
        package = types.ModuleType("diffusers")
        package.__path__ = [str(root)]
        package.__package__ = "diffusers"
        package.__version__ = importlib.metadata.version("diffusers")
        sys.modules["diffusers"] = package
    elif not hasattr(package, "__path__"):
        raise ImportError("Existing diffusers module is not a package")

    schedulers = sys.modules.get("diffusers.schedulers")
    if schedulers is None:
        schedulers = types.ModuleType("diffusers.schedulers")
        schedulers.__path__ = [str(root / "schedulers")]
        schedulers.__package__ = "diffusers.schedulers"
        sys.modules["diffusers.schedulers"] = schedulers


def _load_module(name: str, path: Path, is_package: bool = False):
    module = sys.modules.get(name)
    if module is not None:
        return module
    kwargs = {}
    if is_package:
        kwargs["submodule_search_locations"] = [str(path.parent)]
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_diffusers_components():
    """Return ``(DDPMScheduler, DDIMScheduler, EMAModel)`` safely."""

    root = _diffusers_root()
    _ensure_lightweight_package(root)

    _load_module("diffusers.utils", root / "utils" / "__init__.py", is_package=True)
    _load_module("diffusers.configuration_utils", root / "configuration_utils.py")
    _load_module(
        "diffusers.schedulers.scheduling_utils",
        root / "schedulers" / "scheduling_utils.py",
    )
    ddpm = _load_module(
        "diffusers.schedulers.scheduling_ddpm",
        root / "schedulers" / "scheduling_ddpm.py",
    )
    ddim = _load_module(
        "diffusers.schedulers.scheduling_ddim",
        root / "schedulers" / "scheduling_ddim.py",
    )
    training = _load_module("diffusers.training_utils", root / "training_utils.py")
    return ddpm.DDPMScheduler, ddim.DDIMScheduler, training.EMAModel
