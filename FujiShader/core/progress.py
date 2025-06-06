# FujiShader/core/progress.py
from __future__ import annotations
from typing import Protocol, Optional

class ProgressReporter(Protocol):
    """進捗レポーターの共通インターフェース（任意実装）"""
    def set_range(self, maximum: int) -> None: ...
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
    def done(self) -> None: ...

class NullProgress:
    """何もしない進捗レポーター（デフォルト用）"""
    def set_range(self, maximum: int) -> None: 
        pass
    
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: 
        pass
    
    def done(self) -> None: 
        pass

# ---- CLI用 -----------------------------------------
class TqdmProgress:
    """コマンドライン用の進捗バー（tqdm使用）"""
    def __init__(self):
        try:
            from tqdm import tqdm      # 遅延import
            self._tqdm_cls = tqdm
            self._available = True
        except ImportError:
            self._available = False
        self._bar = None

    def set_range(self, maximum: int) -> None:
        if self._available:
            self._bar = self._tqdm_cls(
                total=maximum, 
                unit="処理", 
                desc="進捗",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )

    def advance(self, step: int = 1, text: Optional[str] = None) -> None:
        if not self._available or self._bar is None:
            return
        self._bar.update(step)
        if text:
            # 進捗説明テキストを設定（短縮版）
            short_text = text[:50] + "..." if len(text) > 50 else text
            self._bar.set_description_str(short_text)

    def done(self) -> None:
        if self._bar:
            self._bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()

# ---- QGIS用 ----------------------------------------
class QgisProgress:
    """QGIS用進捗レポーター（QgsProcessingFeedback / QgsTask共用）"""
    def __init__(self, feedback):
        self.fb = feedback
        self._max = 100
        self._current = 0  # 現在のステップ数を追跡
        self._last_text = ""  # 最後に表示したテキストを保存

    def set_range(self, maximum: int) -> None:
        """進捗の最大値を設定"""
        self._max = maximum       # step→百分率に換算
        self._current = 0         # リセット

    def advance(self, step: int = 1, text: Optional[str] = None) -> None:
        """進捗を進める"""
        try:
            # キャンセルチェック（利用可能な場合）
            if hasattr(self.fb, 'isCanceled') and self.fb.isCanceled():
                raise RuntimeError("処理がユーザーによってキャンセルされました")
            
            self._current += step
            pct = min(100, (self._current * 100 / self._max)) if self._max > 0 else 100
            
            # 進捗率を設定
            if hasattr(self.fb, 'setProgress'):
                self.fb.setProgress(pct)
            
            # テキストが変更された場合のみ更新
            if text and text != self._last_text:
                self._last_text = text
                # テキストを短縮（QGIS UIの制限を考慮）
                display_text = text[:100] + "..." if len(text) > 100 else text
                
                # ProcessingAlgorithm: pushInfo, Task: setDescription
                if hasattr(self.fb, "pushInfo"):
                    self.fb.pushInfo(display_text)
                elif hasattr(self.fb, "setDescription"):
                    self.fb.setDescription(display_text)
                    
        except Exception as e:
            # QGISフィードバックオブジェクトの問題に対する防御的処理
            # エラーを無視して処理を継続（ログ出力は避ける）
            pass

    def done(self) -> None:
        """進捗完了"""
        try:
            if hasattr(self.fb, 'setProgress'):
                self.fb.setProgress(100)
            
            # 完了メッセージを表示
            completion_text = "処理完了"
            if hasattr(self.fb, "setDescription"):
                self.fb.setDescription(completion_text)
            elif hasattr(self.fb, "pushInfo"):
                self.fb.pushInfo(completion_text)
                
        except Exception:
            # エラーを無視して処理を継続
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()

# ---- コンソール用シンプル進捗 ----------------------
class ConsoleProgress:
    """シンプルなコンソール進捗表示（tqdm不要）"""
    def __init__(self, show_percentage: bool = True):
        self.show_percentage = show_percentage
        self._max = 100
        self._current = 0
        self._last_percent = -1

    def set_range(self, maximum: int) -> None:
        self._max = maximum
        self._current = 0
        self._last_percent = -1

    def advance(self, step: int = 1, text: Optional[str] = None) -> None:
        self._current += step
        
        if self.show_percentage and self._max > 0:
            percent = int((self._current * 100) / self._max)
            # 5%刻みで表示を更新
            if percent >= self._last_percent + 5 or percent >= 100:
                if text:
                    print(f"[{percent:3d}%] {text}")
                else:
                    print(f"[{percent:3d}%] 処理中...")
                self._last_percent = percent
        elif text:
            print(f"[進捗] {text}")

    def done(self) -> None:
        if self.show_percentage:
            print("[100%] 処理完了")
        else:
            print("[完了] 処理完了")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()

# ---- ファクトリー関数 ------------------------------
def create_progress_reporter(
    context: str = "auto", 
    feedback=None,
    verbose: bool = True
) -> ProgressReporter:
    """
    適切な進捗レポーターを自動選択して作成する
    
    Args:
        context: "qgis", "cli", "console", "auto" のいずれか
        feedback: QGIS用のフィードバックオブジェクト
        verbose: 詳細表示を有効にするかどうか
        
    Returns:
        適切な進捗レポーター実装
    """
    if not verbose:
        return NullProgress()
    
    if context == "qgis" or feedback is not None:
        if feedback is None:
            raise ValueError("QGISコンテキストではfeedbackオブジェクトが必要です")
        return QgisProgress(feedback)
    
    if context == "cli" or context == "auto":
        try:
            return TqdmProgress()
        except ImportError:
            if context == "cli":
                print("[警告] tqdmが利用できません。シンプルな進捗表示を使用します。")
            return ConsoleProgress()
    
    if context == "console":
        return ConsoleProgress()
    
    # デフォルト
    return NullProgress()