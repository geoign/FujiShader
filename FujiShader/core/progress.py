# FujiShader/core/progress.py
from __future__ import annotations
from typing import Protocol, Optional

class ProgressReporter(Protocol):
    """共通インターフェース（任意実装）"""
    def set_range(self, maximum: int) -> None: ...
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
    def done(self) -> None: ...

class NullProgress:
    """何もしない（デフォルト用）"""
    def set_range(self, maximum: int) -> None: 
        pass
    
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: 
        pass
    
    def done(self) -> None: 
        pass

# ---- CLI 用 -----------------------------------------
class TqdmProgress:
    def __init__(self):
        try:
            from tqdm import tqdm      # 遅延 import
            self._tqdm_cls = tqdm
            self._available = True
        except ImportError:
            self._available = False
        self._bar = None

    def set_range(self, maximum: int) -> None:
        if self._available:
            self._bar = self._tqdm_cls(total=maximum, unit="tile")

    def advance(self, step: int = 1, text: Optional[str] = None) -> None:
        if not self._available or self._bar is None:
            return
        self._bar.update(step)
        if text:
            self._bar.set_description_str(text)

    def done(self) -> None:
        if self._bar:
            self._bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()

# ---- QGIS 用 ----------------------------------------
class QgisProgress:
    """QgsProcessingFeedback / QgsTask 共用"""
    def __init__(self, feedback):
        self.fb = feedback
        self._max = 100
        self._current = 0  # 現在のステップ数を追跡

    def set_range(self, maximum: int) -> None:
        self._max = maximum       # step→百分率に換算
        self._current = 0         # リセット

    def advance(self, step: int = 1, text: Optional[str] = None) -> None:
        try:
            self._current += step
            pct = min(100, (self._current * 100 / self._max))
            self.fb.setProgress(pct)
            if text:
                # ProcessingAlgorithm: pushInfo, Task: setDescription
                if hasattr(self.fb, "pushInfo"):
                    self.fb.pushInfo(text)
                elif hasattr(self.fb, "setDescription"):
                    self.fb.setDescription(text)
        except Exception:
            # QGISフィードバックオブジェクトの問題に対する防御的処理
            pass

    def done(self) -> None:
        try:
            self.fb.setProgress(100)
            # プログレステキストもクリア（可能であれば）
            if hasattr(self.fb, "setDescription"):
                self.fb.setDescription("")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()