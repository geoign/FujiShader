"""
FujiShader.shader
=================
Dynamic aggregator for all shading / terrain-analysis algorithms
contained in this sub-package.

* Any .py file whose name **doesn’t start with “_”** is imported.
* Every public symbol declared in its ``__all__`` is
  - injected into the shader namespace   →  ``import FujiShader.shader as fs``
  - listed in ``list_algorithms()`` for GUI / CLI discovery.

Add a new algorithm file (e.g. openness.py with ``__all__ = ["openness"]``)
and it will appear automatically—no edit here required.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable, Dict

__all__: list[str] = []          # re-exported callables
_algorithms: Dict[str, Callable] = {}
_modules: Dict[str, ModuleType] = {}

# ---------------------------------------------------------------------------
# auto-import every peer module whose filename doesn't start with "_"
# ---------------------------------------------------------------------------
for modinfo in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
    if modinfo.ispkg or modinfo.name.rsplit('.', 1)[-1].startswith("_"):
        continue  # skip sub-packages & private helpers

    module = importlib.import_module(modinfo.name)
    mod_short = module.__name__.split(".")[-1]
    _modules[mod_short] = module

    # collect public symbols declared in submodule's __all__
    for sym in getattr(module, "__all__", []):
        obj = getattr(module, sym)
        globals()[sym] = obj          # re-export at package level
        __all__.append(sym)
        _algorithms[sym] = obj

# ---------------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------------
def list_algorithms() -> Dict[str, Callable]:
    """
    Return a *copy* of the {name: callable} registry.

    Example
    -------
    >>> import FujiShader.shader as fs
    >>> fs.list_algorithms().keys()
    dict_keys(['topo_usm', 'multi_scale_usm',
               'warmcool_map',
               'slope',
               'skyview_factor'])
    """
    return _algorithms.copy()


def list_modules() -> list[str]:
    """Return the short names of the sub-modules currently loaded."""
    return sorted(_modules.keys())
