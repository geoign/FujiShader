"""
FujiShader.shader
=================
このサブパッケージに含まれるすべてのシェーディング/地形解析アルゴリズムの
動的アグリゲーター。

* ファイル名が**"_"で始まらない**すべての.pyファイルが自動インポートされます。
* ``__all__``で宣言されたすべての公開シンボルは以下のように処理されます：
  - shaderネームスペースに注入される   →  ``import FujiShader.shader as fs``
  - GUI/CLI発見用の``list_algorithms()``にリストされる

新しいアルゴリズムファイル（例：``__all__ = ["openness"]``のopenness.py）を
追加すると、自動的に表示されます—ここでの編集は不要です。
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable, Dict

__all__: list[str] = []          # 再エクスポートされる呼び出し可能オブジェクト
_algorithms: Dict[str, Callable] = {}
_modules: Dict[str, ModuleType] = {}

# ---------------------------------------------------------------------------
# ファイル名が"_"で始まらないすべてのピアモジュールを自動インポート
# ---------------------------------------------------------------------------
for modinfo in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
    if modinfo.ispkg or modinfo.name.rsplit('.', 1)[-1].startswith("_"):
        continue  # サブパッケージとプライベートヘルパーをスキップ

    module = importlib.import_module(modinfo.name)
    mod_short = module.__name__.split(".")[-1]
    _modules[mod_short] = module

    # サブモジュールの__all__で宣言された公開シンボルを収集
    for sym in getattr(module, "__all__", []):
        obj = getattr(module, sym)
        globals()[sym] = obj          # パッケージレベルで再エクスポート
        __all__.append(sym)
        _algorithms[sym] = obj

# ---------------------------------------------------------------------------
# ヘルパーユーティリティ
# ---------------------------------------------------------------------------
def list_algorithms() -> Dict[str, Callable]:
    """
    {名前: 呼び出し可能オブジェクト}レジストリの*コピー*を返します。

    例
    -------
    >>> import FujiShader.shader as fs
    >>> fs.list_algorithms().keys()
    dict_keys(['topo_usm', 'multi_scale_usm',
               'warmcool_map',
               'slope',
               'skyview_factor',
               'ambient_occlusion'])
    """
    return _algorithms.copy()


def list_modules() -> list[str]:
    """現在読み込まれているサブモジュールの短縮名を返します。"""
    return sorted(_modules.keys())