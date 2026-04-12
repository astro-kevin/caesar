from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _matching_extension_paths() -> list[Path]:
    caesar_dir = Path(__file__).resolve().parent
    candidates = sorted(caesar_dir.glob("group_funcs*.so"))
    if not candidates:
        return []

    abi_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    tagged = [path for path in candidates if abi_tag in path.name]
    return tagged or candidates


@lru_cache(maxsize=1)
def _load_group_funcs_extension() -> ModuleType:
    caesar_dir = Path(__file__).resolve().parent
    errors = []
    try:
        importlib.invalidate_caches()
        if str(caesar_dir) not in sys.path:
            sys.path.insert(0, str(caesar_dir))
        module = importlib.import_module("group_funcs")
        if getattr(module, "__file__", None):
            return module
    except Exception as exc:
        errors.append(f"import group_funcs: {exc}")

    for path in _matching_extension_paths():
        try:
            spec = importlib.util.spec_from_file_location("group_funcs", str(path))
            if spec is None or spec.loader is None:
                errors.append(f"{path.name}: no loader available")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise ImportError(
            "Unable to load a compatible CAESAR group_funcs extension; "
            + "; ".join(errors)
        )
    raise ImportError("No CAESAR group_funcs extension was found next to the local package")


def load_group_funcs(*names: str):
    module = None
    try:
        module = importlib.import_module("caesar.group_funcs")
    except Exception:
        module = None

    if module is None or any(not hasattr(module, name) for name in names):
        module = _load_group_funcs_extension()

    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "CAESAR group_funcs is missing required symbols: " + ", ".join(sorted(missing))
        )

    if not names:
        return module
    return tuple(getattr(module, name) for name in names)
