import argparse
import faulthandler
import hashlib
import json
import logging
import os
import sys
import threading
import traceback
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets



APP_NAME = "CSV\u591a\u4fe1\u53f7\u6ce2\u5f62\u67e5\u770b\u5668"
_LOGGER: Optional[logging.Logger] = None
_LOG_FILE_PATH: Optional[str] = None
_LAST_ERROR_SHOWN = {"active": False, "last_ts": 0.0}


def _get_log_file_path() -> str:
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH is not None:
        return _LOG_FILE_PATH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE_PATH = os.path.join(log_dir, f"runtime_{ts}.log")
    return _LOG_FILE_PATH


def setup_runtime_diagnostics() -> str:
    global _LOGGER
    log_path = _get_log_file_path()
    logger = logging.getLogger("csv_wave_viewer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    _LOGGER = logger
    logger.info("==== \u7a0b\u5e8f\u542f\u52a8 ====")
    logger.info("Python=%s", sys.version.replace("\n", " "))
    logger.info("argv=%s", sys.argv)
    try:
        crash_path = os.path.join(os.path.dirname(log_path), "crash_dump.log")
        crash_file = open(crash_path, "a", encoding="utf-8")
        faulthandler.enable(file=crash_file, all_threads=True)
        logger.info("faulthandler enabled: %s", crash_path)
    except Exception as exc:
        logger.warning("faulthandler enable failed: %s", exc)
    return log_path


def _log_exception_block(title: str, detail: str) -> None:
    if _LOGGER is not None:
        _LOGGER.error("%s\n%s", title, detail)
    else:
        print(f"{title}\n{detail}", file=sys.stderr)


def _show_runtime_error_dialog(title: str, detail: str) -> None:
    import time

    now = time.time()
    if now - float(_LAST_ERROR_SHOWN.get("last_ts", 0.0)) < 2.0:
        return
    if _LAST_ERROR_SHOWN["active"]:
        return

    _LAST_ERROR_SHOWN["last_ts"] = now
    _LAST_ERROR_SHOWN["active"] = True
    try:
        log_path = _get_log_file_path()
        msg = (
            f"{title}\n\n"
            f"\u7a0b\u5e8f\u5df2\u8bb0\u5f55\u65e5\u5fd7\uff0c\u8def\u5f84\uff1a\n{log_path}\n\n"
            "\u8bf7\u628a\u8be5\u65e5\u5fd7\u6587\u4ef6\u6216\u622a\u56fe\u53d1\u7ed9\u6211\uff0c\u6211\u53ef\u4ee5\u5feb\u901f\u5b9a\u4f4d\u3002"
        )
        QtWidgets.QMessageBox.critical(None, APP_NAME + " \u8fd0\u884c\u9519\u8bef", msg)
    except Exception:
        pass
    finally:
        _LAST_ERROR_SHOWN["active"] = False


def install_exception_hooks() -> None:
    def should_suppress(exc_type, detail: str) -> bool:
        # pyqtgraph????????????????????????????
        if exc_type is AttributeError and "autoRangeEnabled" in detail and "GraphicsLayoutWidget" in detail:
            return True
        return False

    def handle_sys_exception(exc_type, exc_value, exc_tb):
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        if should_suppress(exc_type, detail):
            if _LOGGER is not None:
                _LOGGER.warning("\u5df2\u6291\u5236\u5df2\u77e5\u5185\u90e8\u5f02\u5e38: %s", detail.splitlines()[-1] if detail else "")
            return
        _log_exception_block("\u672a\u6355\u83b7\u5f02\u5e38(sys.excepthook)", detail)
        _show_runtime_error_dialog("\u53d1\u751f\u672a\u6355\u83b7\u5f02\u5e38", detail)

    def handle_thread_exception(args):
        detail = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        _log_exception_block(f"\u7ebf\u7a0b\u5f02\u5e38({args.thread.name})", detail)
        _show_runtime_error_dialog("\u540e\u53f0\u7ebf\u7a0b\u53d1\u751f\u5f02\u5e38", detail)

    def handle_qt_message(mode, context, message):
        if mode == QtCore.QtFatalMsg:
            level = "FATAL"
        elif mode == QtCore.QtCriticalMsg:
            level = "CRITICAL"
        elif mode == QtCore.QtWarningMsg:
            level = "WARNING"
        elif mode == QtCore.QtInfoMsg:
            level = "INFO"
        else:
            level = "DEBUG"
        if _LOGGER is not None:
            _LOGGER.info("Qt[%s] %s", level, message)
        if mode in (QtCore.QtFatalMsg, QtCore.QtCriticalMsg):
            _show_runtime_error_dialog(f"Qt\u8fd0\u884c\u6d88\u606f: {level}", str(message))

    sys.excepthook = handle_sys_exception
    try:
        threading.excepthook = handle_thread_exception
    except Exception:
        pass
    QtCore.qInstallMessageHandler(handle_qt_message)


class ConciseDateAxisItem(pg.DateAxisItem):
    """简洁时间轴：同日显示时分，跨日或关键刻度补充日期。"""

    def tickStrings(self, values, scale, spacing):
        try:
            spacing = float(spacing)
        except Exception:
            spacing = 0.0

        out = []
        prev_day = None
        for v in values:
            try:
                dt = pd.to_datetime(float(v), unit="s", utc=True).to_pydatetime()
            except Exception:
                out.append("")
                continue

            day = dt.date()
            if spacing >= 86400:  # 天级
                text = dt.strftime("%m-%d")
            elif spacing >= 3600:  # 小时级
                if prev_day is None or day != prev_day:
                    text = dt.strftime("%m-%d %H:%M")
                else:
                    text = dt.strftime("%H:%M")
            else:  # 分钟/秒级
                if prev_day is None or day != prev_day:
                    text = dt.strftime("%m-%d %H:%M")
                else:
                    text = dt.strftime("%H:%M:%S")
            out.append(text)
            prev_day = day
        return out


def detect_time_column(columns: List[str]) -> Optional[str]:
    keywords = ("时间", "time", "timestamp", "date", "日期")
    for col in columns:
        lower_col = str(col).lower()
        if any(k in lower_col for k in keywords) or "时间" in str(col) or "日期" in str(col):
            return col
    return None


def try_parse_time_seconds(series: pd.Series) -> Optional[np.ndarray]:
    def datetime_series_to_seconds(dt: pd.Series) -> np.ndarray:
        unit = np.datetime_data(dt.dtype)[0]
        scale = {
            "ns": 1_000_000_000.0,
            "us": 1_000_000.0,
            "ms": 1_000.0,
            "s": 1.0,
            "m": 1.0 / 60.0,
            "h": 1.0 / 3600.0,
            "D": 1.0 / 86400.0,
        }.get(unit)
        if scale is None:
            return np.full(len(dt), np.nan, dtype=np.float64)
        ints = dt.astype("int64").to_numpy(dtype=np.float64)
        return ints / scale

    # 先尝试字符串日期时间
    text = series.astype(str).str.strip().str.rstrip(".").str.replace("T", " ", regex=False)
    parsed_text = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    if parsed_text.notna().mean() < 0.6:
        parsed_text = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    text_valid = parsed_text.notna().mean()
    if text_valid >= 0.6:
        x_sec = np.full(len(series), np.nan, dtype=np.float64)
        valid = parsed_text.notna()
        x_sec[valid.to_numpy()] = datetime_series_to_seconds(parsed_text[valid])
        return x_sec

    # 再尝试数值时间
    num = pd.to_numeric(series, errors="coerce")
    valid = num.notna()
    if valid.mean() < 0.6:
        return None

    num_vals = num.to_numpy(dtype=np.float64)
    finite = num_vals[np.isfinite(num_vals)]
    if finite.size == 0:
        return None

    # 常见紧凑日期格式：20260415110809 / 20260415110809123
    as_text = pd.Series("", index=series.index, dtype="object")
    as_text.loc[valid] = num.loc[valid].round().astype(np.int64).astype(str)
    lens = as_text.loc[valid].str.len()
    if not lens.empty:
        mode_len = int(lens.mode().iat[0])
        mode_ratio = float((lens == mode_len).sum()) / float(len(lens))
        if mode_len in (14, 17) and mode_ratio >= 0.85:
            fmt = "%Y%m%d%H%M%S" if mode_len == 14 else "%Y%m%d%H%M%S%f"
            parsed_num_date = pd.to_datetime(as_text, format=fmt, errors="coerce")
            if parsed_num_date.notna().mean() >= 0.6:
                x_sec = np.full(len(series), np.nan, dtype=np.float64)
                ok = parsed_num_date.notna()
                x_sec[ok.to_numpy()] = datetime_series_to_seconds(parsed_num_date[ok])
                return x_sec

    # epoch 推断
    med = float(np.nanmedian(np.abs(finite)))
    if med >= 1e17:
        x_sec = num_vals / 1_000_000_000.0  # ns -> s
    elif med >= 1e14:
        x_sec = num_vals / 1_000_000.0  # us -> s
    elif med >= 1e11:
        x_sec = num_vals / 1000.0  # ms -> s
    elif med >= 1e9:
        x_sec = num_vals  # s
    else:
        return None
    return x_sec


def monotonic_ratio(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    if vals.size < 3:
        return 0.0
    d = np.diff(vals)
    return float((d >= 0).sum()) / float(len(d))


def detect_best_time_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    by_name = detect_time_column(list(df.columns))
    named_candidates = []
    for col in df.columns:
        lower_col = str(col).lower()
        if any(k in lower_col for k in ("时间", "日期", "time", "timestamp", "date")):
            named_candidates.append(col)

    candidate_cols = named_candidates if named_candidates else list(df.columns[:40])
    candidates = []
    for col in candidate_cols:
        parsed = try_parse_time_seconds(df[col])
        if parsed is None:
            continue
        valid_ratio = float(np.isfinite(parsed).mean())
        mono = monotonic_ratio(parsed)
        vals = parsed[np.isfinite(parsed)]
        abs_time_bonus = 0.0
        if vals.size:
            med = float(np.nanmedian(vals))
            if 946684800 <= med <= 4102444800:  # 2000-01-01 到 2100-01-01
                abs_time_bonus = 0.15
        name_bonus = 0.08 if by_name is not None and col == by_name else 0.0
        score = valid_ratio * 0.55 + mono * 0.22 + name_bonus + abs_time_bonus
        candidates.append((score, valid_ratio, mono, col))

    if not candidates:
        return by_name

    candidates.sort(reverse=True)
    best = candidates[0]
    # 需要基本可用才选为时间列
    if best[1] >= 0.6 and best[2] >= 0.7:
        return best[3]
    return by_name


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    encodings = ["gbk", "utf-8-sig", "utf-8"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_error = exc
    raise RuntimeError(f"读取CSV失败，尝试编码 {encodings} 都不成功。最后错误: {last_error}")


class CsvLoadWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(object, object, str, str)
    error = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(self, path: str, forced_time_col: Optional[str] = None):
        super().__init__()
        self.path = path
        self.forced_time_col = forced_time_col
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _cache_paths(self) -> Tuple[str, str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, "_data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        abs_path = os.path.abspath(self.path)
        key = hashlib.sha1(abs_path.encode("utf-8", errors="ignore")).hexdigest()
        return (
            os.path.join(cache_dir, f"{key}.pkl"),
            os.path.join(cache_dir, f"{key}.meta.json"),
        )

    def _load_from_cache_if_valid(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        data_path, meta_path = self._cache_paths()
        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            return None, None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st = os.stat(self.path)
            if int(meta.get("source_size", -1)) != int(st.st_size):
                return None, None
            if int(meta.get("source_mtime_ns", -1)) != int(st.st_mtime_ns):
                return None, None

            self.progress.emit(8, "命中本地缓存，正在读取...")
            df = pd.read_pickle(data_path)
            cached_time_col = meta.get("time_col")
            if isinstance(cached_time_col, str) and cached_time_col in df.columns:
                time_col = cached_time_col
            else:
                time_col = None
            return df, time_col
        except Exception:
            return None, None

    def _write_cache_best_effort(self, df: pd.DataFrame, time_col: Optional[str]) -> None:
        try:
            data_path, meta_path = self._cache_paths()
            st = os.stat(self.path)
            df.to_pickle(data_path)
            meta = {
                "schema_version": 1,
                "source_path": os.path.abspath(self.path),
                "source_size": int(st.st_size),
                "source_mtime_ns": int(st.st_mtime_ns),
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "time_col": time_col if isinstance(time_col, str) else None,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _count_lines(self) -> int:
        size = max(1, os.path.getsize(self.path))
        line_count = 0
        read_size = 0
        with open(self.path, "rb") as f:
            while True:
                if self._cancel_requested:
                    return 0
                block = f.read(4 * 1024 * 1024)
                if not block:
                    break
                line_count += block.count(b"\n")
                read_size += len(block)
                ratio = read_size / size
                self.progress.emit(min(8, int(1 + ratio * 7)), "正在统计数据规模...")
        return max(1, line_count)

    def run(self) -> None:
        self.progress.emit(1, "检查本地缓存...")
        cached_df, cached_time_col = self._load_from_cache_if_valid()
        if cached_df is not None:
            if self._cancel_requested:
                self.canceled.emit()
                return

            time_col = self.forced_time_col
            if time_col is not None and time_col not in cached_df.columns:
                time_col = None
            if time_col is None:
                time_col = cached_time_col
            if time_col is None:
                time_col = detect_best_time_column(cached_df)

            self.progress.emit(100, "缓存加载完成")
            self.finished.emit(cached_df, time_col, os.path.basename(self.path), "cache_hit")
            return

        encodings = ["gbk", "utf-8-sig", "utf-8"]
        errors = []

        for enc in encodings:
            if self._cancel_requested:
                self.canceled.emit()
                return

            try:
                self.progress.emit(1, f"尝试编码 {enc}...")
                total_lines = self._count_lines()
                if self._cancel_requested:
                    self.canceled.emit()
                    return

                total_rows = max(1, total_lines - 1)
                rows_read = 0
                chunks = []

                reader = pd.read_csv(self.path, encoding=enc, chunksize=50000)
                for chunk in reader:
                    if self._cancel_requested:
                        self.canceled.emit()
                        return

                    chunks.append(chunk)
                    rows_read += len(chunk)
                    ratio = min(1.0, rows_read / total_rows)
                    pct = 10 + int(ratio * 80)
                    self.progress.emit(pct, f"读取中... {rows_read:,}/{total_rows:,} 行")

                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.DataFrame()

                self.progress.emit(93, "识别时间列...")
                time_col = self.forced_time_col
                if time_col is not None and time_col not in df.columns:
                    time_col = None
                if time_col is None:
                    time_col = detect_best_time_column(df)

                self.progress.emit(96, "写入本地缓存...")
                self._write_cache_best_effort(df, time_col)
                self.progress.emit(100, "加载完成")
                self.finished.emit(df, time_col, os.path.basename(self.path), "csv_loaded")
                return
            except Exception as exc:
                errors.append(f"{enc}: {exc}")

        self.error.emit("读取失败:\n" + "\n".join(errors))


class CsvWaveViewer(QtWidgets.QMainWindow):
    def __init__(self, df: pd.DataFrame, time_col: Optional[str], title: str = "CSV多信号波形查看器"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1500, 900)

        self.df = pd.DataFrame()
        self.time_col: Optional[str] = None
        self.x_raw = np.array([])
        self.x_display = np.array([])
        self.is_time_axis = False
        self.numeric_cols: List[str] = []

        self.left_panel = QtWidgets.QFrame()
        self.plot_widget = self._build_plot_area()
        self.value_panel = self._build_value_panel()
        self.plot_value_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.plot_value_splitter.addWidget(self.plot_widget)
        self.plot_value_splitter.addWidget(self.value_panel)
        self.plot_value_splitter.setChildrenCollapsible(False)
        self.plot_value_splitter.setHandleWidth(6)
        self.plot_value_splitter.setStretchFactor(0, 1)
        self.plot_value_splitter.setStretchFactor(1, 0)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.left_panel, 0)
        layout.addWidget(self.plot_value_splitter, 1)
        self.setCentralWidget(central)
        self.plot_value_splitter.setSizes([980, 520])

        self.plot_items = []
        self.plot_cols: List[str] = []
        self.curves = {}
        self.vlines = []
        self.overview_plot = None
        self.top_axis_plot = None
        self.hover_proxy = None
        self.selected_cols: List[str] = []
        self._syncing_from_region = False
        self._syncing_from_plot = False
        self.favorites_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favorites.json")
        self.favorites = self._read_favorites()
        self.marker_snapshots = [None, None, None]
        self.marker_next_slot = 0
        self.marker_colors = [(230, 80, 80), (80, 150, 240), (70, 170, 90)]
        self.marker_slot_lines = [[], [], []]
        self.marker_bottom_label_items = [None, None, None]
        self.marker_bottom_xs = None
        self.marker_delete_targets = []
        self._menu_localizing = False
        self._scene_obj = None
        self._drag_select_candidate = False
        self._drag_select_active = False
        self._drag_start_scene_pos = None
        self._drag_start_x = None
        self._drag_active_plot = None
        self._drag_preview_region = None
        self._drag_threshold_px = 6.0
        self._suppress_next_click = False
        self._hover_gap_threshold = 0.0
        self._col_numeric_cache = {}
        self._col_block_absmax_cache = {}
        self._absmax_block_size = 512
        self._large_data_mode = False
        self._marker_auto_threshold = 3000
        self._curve_marker_visible = {}
        self._y_range_timer = QtCore.QTimer(self)
        self._y_range_timer.setSingleShot(True)
        self._y_range_timer.setInterval(40)
        self._y_range_timer.timeout.connect(self._update_all_y_ranges)

        self.load_thread = None
        self.load_worker = None
        self.progress_dialog = None

        self._setup_top_toolbar()
        self._set_data(df, time_col)
        self.statusBar().showMessage("就绪")

    def _setup_top_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QtGui.QAction("打开CSV", self)
        open_action.triggered.connect(self._open_csv_dialog)
        toolbar.addAction(open_action)

    def _build_x_axis(self, df: pd.DataFrame, time_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, bool]:
        if time_col is None:
            idx = np.arange(len(df), dtype=np.float64)
            return idx, idx, False

        parsed_sec = try_parse_time_seconds(df[time_col])
        if parsed_sec is None:
            idx = np.arange(len(df), dtype=np.float64)
            return idx, idx, False

        x_sec = pd.Series(parsed_sec).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=np.float64)
        if monotonic_ratio(x_sec) < 0.7:
            idx = np.arange(len(df), dtype=np.float64)
            return idx, idx, False
        return x_sec, x_sec, True

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(360)

        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(8)

        title = QtWidgets.QLabel("信号选择")
        title.setStyleSheet("font-weight: bold;")
        vbox.addWidget(title)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("搜索信号名（支持中文/英文）")
        vbox.addWidget(self.search_edit)

        self.selected_count_label = QtWidgets.QLabel("")
        self.selected_count_label.setStyleSheet("color: #666;")
        vbox.addWidget(self.selected_count_label)

        self.signal_list = QtWidgets.QListWidget()
        self.signal_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for col in self.numeric_cols:
            item = QtWidgets.QListWidgetItem(col)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.signal_list.addItem(item)
        vbox.addWidget(self.signal_list, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("全选")
        self.btn_none = QtWidgets.QPushButton("全不选")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        vbox.addLayout(btn_row)

        fav_title = QtWidgets.QLabel("收藏")
        fav_title.setStyleSheet("font-weight: bold;")
        vbox.addWidget(fav_title)

        self.fav_combo = QtWidgets.QComboBox()
        self._refresh_favorites_combo()
        vbox.addWidget(self.fav_combo)

        fav_btn_row1 = QtWidgets.QHBoxLayout()
        self.btn_fav_save = QtWidgets.QPushButton("收藏当前")
        self.btn_fav_apply = QtWidgets.QPushButton("应用收藏")
        fav_btn_row1.addWidget(self.btn_fav_save)
        fav_btn_row1.addWidget(self.btn_fav_apply)
        vbox.addLayout(fav_btn_row1)

        fav_btn_row2 = QtWidgets.QHBoxLayout()
        self.btn_fav_delete = QtWidgets.QPushButton("删除收藏")
        fav_btn_row2.addWidget(self.btn_fav_delete)
        vbox.addLayout(fav_btn_row2)

        tips = QtWidgets.QLabel(
            "使用说明:\n"
            "1) 在左侧搜索并勾选信号（默认不勾选）\n"
            "2) 勾选一个就会在右侧新增一个波形\n"
            "3) 底部总览图拖拽灰色区域选择时间段\n"
            "4) 鼠标移动到任意子图，右侧显示当前时刻各信号值\n"
            "5) 可将当前勾选信号保存为收藏并一键恢复"
        )
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #555;")
        vbox.addWidget(tips)

        return panel

    def _build_plot_area(self) -> pg.GraphicsLayoutWidget:
        return pg.GraphicsLayoutWidget()

    def _build_value_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(1200)

        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(6)
        self.value_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.value_splitter.setChildrenCollapsible(False)
        self.value_splitter.setHandleWidth(6)

        current_widget = QtWidgets.QWidget()
        current_layout = QtWidgets.QVBoxLayout(current_widget)
        current_layout.setContentsMargins(0, 0, 0, 0)
        current_layout.setSpacing(4)
        title = QtWidgets.QLabel("当前光标数值")
        title.setStyleSheet("font-weight: bold;")
        current_layout.addWidget(title)
        self.value_display = QtWidgets.QPlainTextEdit()
        self.value_display.setReadOnly(True)
        self.value_display.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.value_display.setPlainText("鼠标移动到波形区域后，这里会显示当前时刻各信号值。")
        current_layout.addWidget(self.value_display, 1)
        self.value_splitter.addWidget(current_widget)

        self.marker_time_labels = []
        self.marker_text_edits = []
        self.marker_clear_buttons = []
        for i in range(3):
            group = QtWidgets.QGroupBox(f"标记点{i + 1}")
            gv = QtWidgets.QVBoxLayout(group)
            gv.setContentsMargins(8, 8, 8, 8)
            gv.setSpacing(4)

            btn_row = QtWidgets.QHBoxLayout()
            btn_row.addStretch(1)
            btn_clear = QtWidgets.QToolButton()
            btn_clear.setText("×")
            btn_clear.setToolTip(f"清除标记点{i + 1}")
            btn_clear.setAutoRaise(True)
            btn_clear.setFixedWidth(22)
            btn_clear.clicked.connect(lambda _checked=False, slot=i: self._clear_marker(slot))
            btn_row.addWidget(btn_clear)
            gv.addLayout(btn_row)

            tlabel = QtWidgets.QLabel("未标记")
            tlabel.setStyleSheet("color: #555;")
            gv.addWidget(tlabel)

            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            edit.setMinimumHeight(120)
            edit.setPlainText("（空）")
            gv.addWidget(edit, 1)

            self.marker_time_labels.append(tlabel)
            self.marker_text_edits.append(edit)
            self.marker_clear_buttons.append(btn_clear)
            self.value_splitter.addWidget(group)

        vbox.addWidget(self.value_splitter, 1)
        self.value_splitter.setSizes([240, 220, 220, 220])

        self.btn_clear_all_markers = QtWidgets.QPushButton("清空全部标记")
        self.btn_clear_all_markers.clicked.connect(self._clear_all_markers)
        vbox.addWidget(self.btn_clear_all_markers, 0)
        return panel

    def _make_axis_items(self) -> Optional[dict]:
        if not self.is_time_axis:
            return None
        # 强制按 UTC 基准显示，避免系统时区导致 +8/-8 小时偏移
        return {"bottom": ConciseDateAxisItem(orientation="bottom", utcOffset=0)}

    def _bind_left_signals(self) -> None:
        self.signal_list.itemChanged.connect(self._on_signal_item_changed)
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.search_edit.textChanged.connect(self._filter_signals)
        self.btn_fav_save.clicked.connect(self._save_current_favorite)
        self.btn_fav_apply.clicked.connect(self._apply_selected_favorite)
        self.btn_fav_delete.clicked.connect(self._delete_selected_favorite)

    def _localize_menu_recursive(self, menu: QtWidgets.QMenu) -> None:
        if self._menu_localizing:
            return
        self._menu_localizing = True
        mapping = {
            "View All": "查看全部",
            "X axis": "X轴",
            "Y axis": "Y轴",
            "Mouse Mode": "鼠标模式",
            "Plot Options": "图形选项",
            "Export...": "导出...",
            "Transforms": "变换",
            "Downsample": "降采样",
            "Average": "平均",
            "Alpha": "透明度",
            "Grid": "网格",
            "Points": "点",
            "Pan": "平移",
            "Zoom": "缩放",
            "3 button": "三键模式",
            "1 button": "单键模式",
            "Link axis": "关联坐标轴",
            "Link Axis": "关联坐标轴",
            "Link X axis": "关联X轴",
            "Link Y axis": "关联Y轴",
            "Auto": "自动",
            "Manual": "手动",
            "Off": "关闭",
            "Value": "数值",
            "Visible": "可见",
            "Auto Pan": "自动平移",
            "Invert Axis": "反转坐标轴",
            "Mouse Enabled": "允许鼠标交互",
        }

        def translate_text(txt: str) -> str:
            raw = txt or ""
            key = raw.strip().replace("&", "")
            if key in mapping:
                return mapping[key]
            for src, dst in mapping.items():
                if key.startswith(src + ":"):
                    return key.replace(src + ":", dst + ":", 1)
            return raw

        try:
            for act in menu.actions():
                txt = act.text()
                new_txt = translate_text(txt)
                if new_txt != txt and new_txt != "":
                    act.setText(new_txt)
                sub = act.menu()
                if sub is not None:
                    self._localize_menu_recursive(sub)
        finally:
            self._menu_localizing = False

    def _install_menu_localizer(self, menu: QtWidgets.QMenu) -> None:
        # 保留接口，当前不再挂动态钩子，避免菜单展开时卡顿
        return

    def _localize_plot_context_menu(self, plot_item) -> None:
        vb = plot_item.getViewBox()
        try:
            menu = vb.getMenu(None)
            if menu is not None:
                self._localize_menu_recursive(menu)
        except Exception:
            pass

    def _set_data(self, df: pd.DataFrame, time_col: Optional[str]) -> None:
        self.df = df
        self.time_col = time_col
        self._col_numeric_cache = {}
        self._col_block_absmax_cache = {}
        self._curve_marker_visible = {}
        self.x_raw, self.x_display, self.is_time_axis = self._build_x_axis(df, time_col)
        self._hover_gap_threshold = self._compute_hover_gap_threshold()

        self.numeric_cols = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not self.numeric_cols:
            raise ValueError("CSV中没有可绘制的数值列。")
        self._update_performance_mode()

        new_left = self._build_left_panel()
        old_left = self.left_panel
        parent_layout = self.centralWidget().layout()
        parent_layout.replaceWidget(old_left, new_left)
        old_left.deleteLater()
        self.left_panel = new_left

        self._bind_left_signals()
        self._update_selected_count()
        self._clear_all_markers(silent=True)
        self._init_plots()

    def _update_performance_mode(self) -> None:
        rows = int(len(self.df))
        cols = int(len(self.numeric_cols))
        est_points = rows * max(1, cols)
        self._large_data_mode = (rows >= 250_000) or (est_points >= 8_000_000)
        pg.setConfigOptions(antialias=(not self._large_data_mode))

    def _get_numeric_column_array(self, col: str) -> np.ndarray:
        cached = self._col_numeric_cache.get(col)
        if cached is not None:
            return cached
        arr = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=np.float64)
        self._col_numeric_cache[col] = arr
        return arr

    def _get_absmax_blocks(self, col: str) -> np.ndarray:
        cached = self._col_block_absmax_cache.get(col)
        if cached is not None:
            return cached

        y = self._get_numeric_column_array(col)
        n = len(y)
        bs = int(self._absmax_block_size)
        if n <= 0 or bs <= 0:
            out = np.array([], dtype=np.float64)
            self._col_block_absmax_cache[col] = out
            return out

        n_blocks = int(np.ceil(float(n) / float(bs)))
        out = np.full(n_blocks, np.nan, dtype=np.float64)
        for bi in range(n_blocks):
            s = bi * bs
            e = min(n, s + bs)
            seg = y[s:e]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                out[bi] = float(np.nanmax(np.abs(seg)))
        self._col_block_absmax_cache[col] = out
        return out

    def _query_absmax_in_range(self, col: str, i0: int, i1: int) -> float:
        y = self._get_numeric_column_array(col)
        n = len(y)
        if n <= 0:
            return np.nan
        i0 = max(0, min(int(i0), n - 1))
        i1 = max(i0 + 1, min(int(i1), n))
        if i1 - i0 <= 4096:
            seg = y[i0:i1]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                return np.nan
            return float(np.nanmax(np.abs(seg)))

        bs = int(self._absmax_block_size)
        blocks = self._get_absmax_blocks(col)
        if blocks.size == 0:
            return np.nan

        first_full = (i0 + bs - 1) // bs
        last_full = i1 // bs
        best = np.nan

        left_edge_end = min(i1, first_full * bs)
        if left_edge_end > i0:
            seg = y[i0:left_edge_end]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                best = float(np.nanmax(np.abs(seg)))

        right_edge_start = max(i0, last_full * bs)
        if i1 > right_edge_start:
            seg = y[right_edge_start:i1]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                cur = float(np.nanmax(np.abs(seg)))
                best = cur if not np.isfinite(best) else max(best, cur)

        if first_full < last_full:
            mid = blocks[first_full:last_full]
            mid = mid[np.isfinite(mid)]
            if mid.size > 0:
                cur = float(np.nanmax(mid))
                best = cur if not np.isfinite(best) else max(best, cur)
        return best

    def _schedule_y_range_update(self) -> None:
        if self._y_range_timer.isActive():
            self._y_range_timer.start()
            return
        self._y_range_timer.start()

    def _update_curve_downsampling(self) -> None:
        if not self.plot_items or not self.curves:
            return

        for p, col in zip(self.plot_items, self.plot_cols):
            curve = self.curves.get(col)
            if curve is None:
                continue
            try:
                x_left, x_right = p.vb.viewRange()[0]
            except Exception:
                x_left, x_right = float(self.x_display[0]), float(self.x_display[-1])
            if x_left > x_right:
                x_left, x_right = x_right, x_left

            i0 = int(np.searchsorted(self.x_display, x_left, side="left"))
            i1 = int(np.searchsorted(self.x_display, x_right, side="right"))
            i0 = max(0, min(i0, len(self.x_display) - 1))
            i1 = max(i0 + 1, min(i1, len(self.x_display)))
            visible_count = max(1, i1 - i0)

            view_px = max(220, int(p.vb.width()))
            target_points = max(1200, view_px * (2 if self._large_data_mode else 3))
            ds = max(1, int(np.ceil(float(visible_count) / float(target_points))))
            try:
                curve.setDownsampling(ds=ds, auto=False, method="peak")
                curve.setClipToView(True)
            except Exception:
                pass
            want_marker = visible_count <= int(self._marker_auto_threshold)
            has_marker = bool(self._curve_marker_visible.get(col, False))
            if want_marker != has_marker:
                try:
                    if want_marker:
                        pen_obj = curve.opts.get("pen")
                        if hasattr(pen_obj, "color"):
                            marker_color = pen_obj.color()
                        else:
                            marker_color = pg.mkColor(220, 220, 220)
                        curve.setSymbol("o")
                        curve.setSymbolSize(4)
                        curve.setSymbolPen(pg.mkPen((255, 255, 255, 120), width=0.6))
                        curve.setSymbolBrush(pg.mkBrush(marker_color))
                    else:
                        curve.setSymbol(None)
                except Exception:
                    pass
                self._curve_marker_visible[col] = want_marker

    def _open_csv_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择CSV文件",
            os.getcwd(),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self.start_load_csv(path)

    def start_load_csv(self, path: str, forced_time_col: Optional[str] = None) -> None:
        if self.load_thread is not None:
            QtWidgets.QMessageBox.information(self, "提示", "当前正在加载，请稍候。")
            return

        self.progress_dialog = QtWidgets.QProgressDialog("正在加载CSV...", "取消", 0, 100, self)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self._cancel_loading)

        self.load_thread = QtCore.QThread(self)
        self.load_worker = CsvLoadWorker(path, forced_time_col)
        self.load_worker.moveToThread(self.load_thread)

        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.progress.connect(self._on_load_progress)
        self.load_worker.finished.connect(self._on_load_finished)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.canceled.connect(self._on_load_canceled)

        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.error.connect(self.load_thread.quit)
        self.load_worker.canceled.connect(self.load_thread.quit)
        self.load_thread.finished.connect(self._cleanup_loader)

        self.load_thread.start()

    def _cancel_loading(self) -> None:
        if self.load_worker is not None:
            self.load_worker.request_cancel()

    def _on_load_progress(self, percent: int, text: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText(text)
            self.progress_dialog.setValue(max(0, min(100, percent)))
        self.statusBar().showMessage(text)

    def _on_load_finished(self, df: pd.DataFrame, time_col: Optional[str], file_name: str, load_source: str = "csv_loaded") -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()

        try:
            self._set_data(df, time_col)
            self.setWindowTitle(f"CSV多信号波形查看器 - {file_name}")
            source_hint = "（缓存）" if load_source == "cache_hit" else ""
            if time_col is None:
                self.statusBar().showMessage(f"加载完成{source_hint}: {file_name}（未识别到时间列，当前为样本轴）")
            else:
                self.statusBar().showMessage(f"加载完成{source_hint}: {file_name}（时间列: {time_col}）")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))

    def _on_load_error(self, message: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.close()
        self.statusBar().showMessage("加载失败")
        QtWidgets.QMessageBox.critical(self, "加载失败", message)

    def _on_load_canceled(self) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.close()
        self.statusBar().showMessage("已取消加载")

    def _cleanup_loader(self) -> None:
        if self.load_worker is not None:
            self.load_worker.deleteLater()
        if self.load_thread is not None:
            self.load_thread.deleteLater()
        self.load_worker = None
        self.load_thread = None
        self.progress_dialog = None

    def _filter_signals(self, text: str) -> None:
        q = text.strip().lower()
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setHidden(q not in item.text().lower())

    def _on_signal_item_changed(self, _item) -> None:
        self._update_selected_count()
        self._init_plots()

    def _update_selected_count(self) -> None:
        selected = 0
        for i in range(self.signal_list.count()):
            if self.signal_list.item(i).checkState() == QtCore.Qt.Checked:
                selected += 1
        self.selected_count_label.setText(f"已选 {selected} / {self.signal_list.count()}")

    def _refresh_favorites_combo(self) -> None:
        if not hasattr(self, "fav_combo") or self.fav_combo is None:
            return
        current = self.fav_combo.currentText().strip()
        self.fav_combo.clear()
        self.fav_combo.addItem("（请选择收藏）")
        for name in sorted(self.favorites.keys()):
            self.fav_combo.addItem(name)
        if current and current in self.favorites:
            self.fav_combo.setCurrentText(current)

    def _read_favorites(self) -> dict:
        if not os.path.exists(self.favorites_path):
            return {}
        try:
            with open(self.favorites_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            cleaned = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    cleaned[k] = [str(x) for x in v]
            return cleaned
        except Exception:
            return {}

    def _write_favorites(self) -> None:
        with open(self.favorites_path, "w", encoding="utf-8") as f:
            json.dump(self.favorites, f, ensure_ascii=False, indent=2)

    def _save_current_favorite(self) -> None:
        cols = self._get_checked_columns()
        if not cols:
            QtWidgets.QMessageBox.information(self, "提示", "请先勾选至少一个信号再收藏。")
            return

        default_name = self.fav_combo.currentText().strip()
        if default_name in ("", "（请选择收藏）"):
            default_name = "我的收藏"

        name, ok = QtWidgets.QInputDialog.getText(self, "收藏当前选择", "请输入收藏名称：", text=default_name)
        if not ok:
            return
        name = name.strip()
        if not name:
            QtWidgets.QMessageBox.information(self, "提示", "收藏名称不能为空。")
            return

        if name in self.favorites:
            reply = QtWidgets.QMessageBox.question(
                self,
                "覆盖确认",
                f"收藏“{name}”已存在，是否覆盖？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        self.favorites[name] = cols
        self._write_favorites()
        self._refresh_favorites_combo()
        self.fav_combo.setCurrentText(name)
        self.statusBar().showMessage(f"已保存收藏: {name}（{len(cols)}个信号）")

    def _apply_selected_favorite(self) -> None:
        name = self.fav_combo.currentText().strip()
        if not name or name == "（请选择收藏）":
            QtWidgets.QMessageBox.information(self, "提示", "请先选择一个收藏。")
            return
        if name not in self.favorites:
            QtWidgets.QMessageBox.information(self, "提示", f"未找到收藏: {name}")
            return

        targets = set(self.favorites[name])
        self.signal_list.blockSignals(True)
        missing = 0
        matched = 0
        existing = {self.signal_list.item(i).text() for i in range(self.signal_list.count())}
        for t in targets:
            if t not in existing:
                missing += 1

        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if item.text() in targets else QtCore.Qt.Unchecked)
            if item.text() in targets:
                matched += 1
        self.signal_list.blockSignals(False)

        self._update_selected_count()
        self._init_plots()
        self.statusBar().showMessage(f"已应用收藏: {name}（匹配{matched}，缺失{missing}）")

    def _delete_selected_favorite(self) -> None:
        name = self.fav_combo.currentText().strip()
        if not name or name == "（请选择收藏）":
            QtWidgets.QMessageBox.information(self, "提示", "请先选择要删除的收藏。")
            return
        if name not in self.favorites:
            QtWidgets.QMessageBox.information(self, "提示", f"未找到收藏: {name}")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "删除确认",
            f"确定删除收藏“{name}”吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        del self.favorites[name]
        self._write_favorites()
        self._refresh_favorites_combo()
        self.statusBar().showMessage(f"已删除收藏: {name}")

    def _init_plots(self) -> None:
        self.plot_widget.clear()
        self.plot_items.clear()
        self.plot_cols.clear()
        self.curves.clear()
        self.vlines.clear()
        self.top_axis_plot = None
        self.marker_slot_lines = [[], [], []]
        self.marker_bottom_label_items = [None, None, None]
        self.marker_bottom_xs = None
        self.marker_delete_targets = []

        checked_cols = self._get_checked_columns()
        self.selected_cols = checked_cols

        if not checked_cols:
            label = self.plot_widget.addLabel("请在左侧勾选信号后显示波形", row=0, col=0)
            label.setText("请在左侧勾选信号后显示波形")
            self.value_display.setPlainText("请先勾选信号。")
            return

        row_offset = 0
        if self.is_time_axis:
            top_axis_items = {"top": ConciseDateAxisItem(orientation="top", utcOffset=0)}
            self.top_axis_plot = self.plot_widget.addPlot(row=0, col=0, title="", axisItems=top_axis_items)
            self.top_axis_plot.hideAxis("left")
            self.top_axis_plot.hideAxis("bottom")
            self.top_axis_plot.showGrid(x=False, y=False)
            self.top_axis_plot.setYRange(0.0, 1.0, padding=0)
            self.top_axis_plot.setMinimumHeight(30)
            self.top_axis_plot.setMaximumHeight(36)
            self.top_axis_plot.setMouseEnabled(x=False, y=False)
            self.top_axis_plot.setMenuEnabled(False)
            self.top_axis_plot.getViewBox().setMouseEnabled(x=False, y=False)
            self.top_axis_plot.getViewBox().setMenuEnabled(False)
            self.top_axis_plot.getViewBox().setLimits(yMin=0.0, yMax=1.0)
            self.top_axis_plot.setLabel("top", "")
            row_offset = 1

        for i, col in enumerate(checked_cols):
            axis_items = self._make_axis_items()
            if axis_items is None:
                p = self.plot_widget.addPlot(row=i + row_offset, col=0, title=str(col))
            else:
                p = self.plot_widget.addPlot(row=i + row_offset, col=0, title=str(col), axisItems=axis_items)

            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", str(col))

            y = self._get_numeric_column_array(col)
            curve = p.plot(
                self.x_display,
                y,
                pen=pg.mkPen(color=pg.intColor(i, hues=max(8, len(checked_cols))), width=2),
                clipToView=True,
                symbol=None,
            )
            self.curves[col] = curve
            self._curve_marker_visible[col] = False

            if i > 0:
                p.setXLink(self.plot_items[0])
            # 上方所有子图统一隐藏X轴刻度与标题，只在底部总览显示时间轴
            p.hideAxis("bottom")

            vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 80, 80, 220), width=2.4))
            p.addItem(vline, ignoreBounds=True)
            self.vlines.append(vline)
            self.plot_items.append(p)
            self.plot_cols.append(col)
            self._localize_plot_context_menu(p)

        overview_row = len(checked_cols) + row_offset
        axis_items = self._make_axis_items()
        if axis_items is None:
            self.overview_plot = self.plot_widget.addPlot(row=overview_row, col=0, title="")
        else:
            self.overview_plot = self.plot_widget.addPlot(
                row=overview_row, col=0, title="", axisItems=axis_items
            )

        # 底部改成“时间进度条”样式：无波形、无Y轴，仅保留时间轴+可拖拽选区
        # 这里提高固定高度，避免在多波形场景下看起来“像消失”。
        self.overview_plot.showGrid(x=False, y=False)
        self.overview_plot.hideAxis("left")
        self.overview_plot.setTitle("")
        self.overview_plot.setLabel("bottom", "时间进度")
        self.overview_plot.setMinimumHeight(84)
        self.overview_plot.setMaximumHeight(108)
        self.overview_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        self.overview_plot.disableAutoRange(axis=pg.ViewBox.XAxis)
        self._localize_plot_context_menu(self.overview_plot)
        self.overview_plot.setYRange(0.0, 1.0, padding=0)

        # 每个标记点在所有子图保留一条固定竖线
        for p in self.plot_items:
            for slot, color in enumerate(self.marker_colors):
                mline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=color, width=1.2))
                mline.hide()
                p.addItem(mline, ignoreBounds=True)
                self.marker_slot_lines[slot].append(mline)

        x_min = float(np.nanmin(self.x_display))
        x_max = float(np.nanmax(self.x_display))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            x_min, x_max = 0.0, max(1.0, float(len(self.x_display) - 1))
        # 时间轴保留左右各1分钟缓冲；非时间轴保持原始边界
        if self.is_time_axis:
            x_min -= 60.0
            x_max += 60.0
        self._x_domain_min = x_min
        self._x_domain_max = x_max
        self.overview_plot.setXRange(x_min, x_max, padding=0)
        self.overview_plot.getViewBox().setLimits(xMin=x_min, xMax=x_max)

        for p in self.plot_items:
            p.getViewBox().setLimits(xMin=x_min, xMax=x_max)
        if self.top_axis_plot is not None:
            self.top_axis_plot.getViewBox().setLimits(xMin=x_min, xMax=x_max, yMin=0.0, yMax=1.0)

        initial_left = x_min + (x_max - x_min) * 0.15
        initial_right = x_min + (x_max - x_min) * 0.65
        self.region = pg.LinearRegionItem(
            values=[initial_left, initial_right],
            brush=pg.mkBrush(100, 100, 180, 70),
            pen=pg.mkPen((70, 70, 140), width=1),
        )
        # 增强左右边界手柄可见性与可命中性，便于拖拽调整窗口宽度
        for edge in self.region.lines:
            edge.setPen(pg.mkPen((72, 72, 150, 230), width=10))
            edge.setHoverPen(pg.mkPen((56, 56, 130, 250), width=14))
        self.region.setZValue(10)
        self.overview_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self._on_region_changed()

        if self.top_axis_plot is not None and self.plot_items:
            self.top_axis_plot.setXLink(self.plot_items[0])
            self.top_axis_plot.setXRange(initial_left, initial_right, padding=0)

        # 在最后一张波形图底部显示“标记点N”（删除入口改到右侧面板）
        if self.plot_items:
            bottom_plot = self.plot_items[-1]
            for slot in range(3):
                txt = pg.TextItem(anchor=(0.5, 1.0), color=self.marker_colors[slot])
                txt.hide()
                bottom_plot.addItem(txt)
                self.marker_bottom_label_items[slot] = txt
        self._refresh_marker_graphics()

        if self.hover_proxy is not None:
            try:
                self.hover_proxy.disconnect()
            except Exception:
                pass
        self.hover_proxy = pg.SignalProxy(self.plot_items[0].scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
        self._install_scene_event_filter(self.plot_items[0].scene())
        try:
            self.plot_items[0].scene().sigMouseClicked.disconnect(self._mouse_clicked)
        except Exception:
            pass
        self.plot_items[0].scene().sigMouseClicked.connect(self._mouse_clicked)
        for p in self.plot_items:
            p.vb.sigXRangeChanged.connect(self._on_main_xrange_changed)
        self._update_curve_downsampling()
        self._update_all_y_ranges()

    def _update_all_y_ranges(self) -> None:
        if not self.plot_items or not hasattr(self, "region"):
            return
        left, right = self.region.getRegion()
        if left > right:
            left, right = right, left
        i0 = int(np.searchsorted(self.x_display, left, side="left"))
        i1 = int(np.searchsorted(self.x_display, right, side="right"))
        i0 = max(0, min(i0, len(self.x_display) - 1))
        i1 = max(i0 + 1, min(i1, len(self.x_display)))

        for p, col in zip(self.plot_items, self.plot_cols):
            peak = self._query_absmax_in_range(col, i0, i1)
            if not np.isfinite(peak):
                continue
            y_max = self._nice_ceil(peak * 1.3)
            y_min = -y_max
            p.setYRange(y_min, y_max, padding=0)

    @staticmethod
    def _nice_ceil(value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            return 1.0
        exp = np.floor(np.log10(value))
        base = 10 ** exp
        ratio = value / base
        if ratio <= 1:
            nice = 1
        elif ratio <= 2:
            nice = 2
        elif ratio <= 5:
            nice = 5
        else:
            nice = 10
        return float(nice * base)

    def _select_all(self) -> None:
        self.signal_list.blockSignals(True)
        for i in range(self.signal_list.count()):
            self.signal_list.item(i).setCheckState(QtCore.Qt.Checked)
        self.signal_list.blockSignals(False)
        self._update_selected_count()
        self._init_plots()

    def _select_none(self) -> None:
        self.signal_list.blockSignals(True)
        for i in range(self.signal_list.count()):
            self.signal_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self.signal_list.blockSignals(False)
        self._update_selected_count()
        self._init_plots()

    def _get_checked_columns(self) -> List[str]:
        cols = []
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                cols.append(item.text())
        return cols

    @staticmethod
    def _clamp_x_window(
        left: float,
        right: float,
        x_min: float,
        x_max: float,
        prefer_width: Optional[float] = None,
    ) -> Tuple[float, float]:
        if left > right:
            left, right = right, left

        span = max(0.0, float(x_max) - float(x_min))
        width = float(right - left)
        if prefer_width is not None and np.isfinite(prefer_width):
            width = float(prefer_width)
        width = max(0.0, min(width, span))

        center = 0.5 * (left + right)
        new_left = center - 0.5 * width
        new_right = center + 0.5 * width

        if new_left < x_min:
            shift = x_min - new_left
            new_left += shift
            new_right += shift
        if new_right > x_max:
            shift = new_right - x_max
            new_left -= shift
            new_right -= shift

        new_left = max(x_min, min(new_left, x_max))
        new_right = max(new_left, min(new_right, x_max))
        return new_left, new_right

    def _on_region_changed(self) -> None:
        if not self.plot_items:
            return
        left, right = self.region.getRegion()
        if hasattr(self, "_x_domain_min") and hasattr(self, "_x_domain_max"):
            x_min = float(self._x_domain_min)
            x_max = float(self._x_domain_max)
            left, right = self._clamp_x_window(left, right, x_min, x_max, prefer_width=(right - left))
            cur_left, cur_right = self.region.getRegion()
            if abs(cur_left - left) > 1e-9 or abs(cur_right - right) > 1e-9:
                self.region.blockSignals(True)
                self.region.setRegion((left, right))
                self.region.blockSignals(False)
        if not self._syncing_from_plot:
            self._syncing_from_region = True
            try:
                for p in self.plot_items:
                    p.setXRange(left, right, padding=0)
            finally:
                self._syncing_from_region = False
        self._update_curve_downsampling()
        self._refresh_marker_graphics()
        self._schedule_y_range_update()

    def _on_main_xrange_changed(self, _vb, ranges) -> None:
        if not self.plot_items or not hasattr(self, "region") or self._syncing_from_region:
            return

        xr = self._extract_xrange(ranges, _vb)
        if xr is None:
            return

        left, right = xr
        if left > right:
            left, right = right, left

        if hasattr(self, "_x_domain_min") and hasattr(self, "_x_domain_max"):
            x_min = float(self._x_domain_min)
            x_max = float(self._x_domain_max)
        else:
            x_min = float(np.nanmin(self.x_display))
            x_max = float(np.nanmax(self.x_display))
        cur_left, cur_right = self.region.getRegion()
        prefer_width = max(0.0, float(cur_right - cur_left))
        left, right = self._clamp_x_window(left, right, x_min, x_max, prefer_width=prefer_width)

        self._syncing_from_plot = True
        try:
            if abs(cur_left - left) > 1e-9 or abs(cur_right - right) > 1e-9:
                self.region.setRegion((left, right))
            self._syncing_from_region = True
            try:
                for p in self.plot_items:
                    p.setXRange(left, right, padding=0)
            finally:
                self._syncing_from_region = False
        finally:
            self._syncing_from_plot = False
        self._update_curve_downsampling()
        self._schedule_y_range_update()

    @staticmethod
    def _extract_xrange(ranges, vb) -> Optional[Tuple[float, float]]:
        try:
            # pyqtgraph may send [xmin, xmax]
            if isinstance(ranges, (list, tuple)) and len(ranges) == 2 and all(
                isinstance(v, (int, float, np.floating)) for v in ranges
            ):
                return float(ranges[0]), float(ranges[1])

            # or [[xmin, xmax], [ymin, ymax]]
            if isinstance(ranges, (list, tuple)) and len(ranges) >= 1:
                xr = ranges[0]
                if isinstance(xr, (list, tuple)) and len(xr) >= 2:
                    return float(xr[0]), float(xr[1])
        except Exception:
            pass

        try:
            xr = vb.viewRange()[0]
            return float(xr[0]), float(xr[1])
        except Exception:
            return None

    def _mouse_moved(self, evt) -> None:
        if not self.plot_items:
            return

        pos = evt[0]
        active_plot = None
        for p in self.plot_items:
            if p.sceneBoundingRect().contains(pos):
                active_plot = p
                break
        if active_plot is None:
            return

        point = active_plot.vb.mapSceneToView(pos)
        x_val = float(point.x())
        self._update_current_value_panel_by_x(x_val)

    def _mouse_clicked(self, evt) -> None:
        if not self.plot_items:
            return
        if self._suppress_next_click:
            self._suppress_next_click = False
            return
        try:
            if evt.button() != QtCore.Qt.LeftButton:
                return
            pos = evt.scenePos()
        except Exception:
            return

        active_plot = None
        for p in self.plot_items:
            if p.sceneBoundingRect().contains(pos):
                active_plot = p
                break
        if active_plot is None:
            return

        point = active_plot.vb.mapSceneToView(pos)
        idx = int(np.searchsorted(self.x_display, float(point.x())))
        idx = max(0, min(idx, len(self.x_display) - 1))
        self._freeze_marker_at_index(idx)

    def _install_scene_event_filter(self, scene_obj) -> None:
        if scene_obj is None:
            return
        if self._scene_obj is scene_obj:
            return
        if self._scene_obj is not None:
            try:
                self._scene_obj.removeEventFilter(self)
            except Exception:
                pass
        self._scene_obj = scene_obj
        self._scene_obj.installEventFilter(self)

    def _clear_drag_preview(self) -> None:
        if self._drag_preview_region is None:
            return
        try:
            if self._drag_active_plot is not None:
                self._drag_active_plot.removeItem(self._drag_preview_region)
        except Exception:
            pass
        self._drag_preview_region = None

    def _find_active_plot_by_scene_pos(self, scene_pos):
        for p in self.plot_items:
            if p.sceneBoundingRect().contains(scene_pos):
                return p
        return None

    def eventFilter(self, obj, event):
        if obj is not self._scene_obj or not self.plot_items:
            return super().eventFilter(obj, event)

        et = event.type()
        if et == QtCore.QEvent.GraphicsSceneWheel:
            if not hasattr(self, "region") or self.overview_plot is None:
                return super().eventFilter(obj, event)
            try:
                pos = event.scenePos()
            except Exception:
                return super().eventFilter(obj, event)
            active_plot = self._find_active_plot_by_scene_pos(pos)
            use_plot = active_plot
            if use_plot is None and self.overview_plot.sceneBoundingRect().contains(pos):
                use_plot = self.overview_plot
            if use_plot is None:
                return super().eventFilter(obj, event)
            if not (hasattr(self, "_x_domain_min") and hasattr(self, "_x_domain_max")):
                return super().eventFilter(obj, event)

            x_min = float(self._x_domain_min)
            x_max = float(self._x_domain_max)
            span = max(1e-12, x_max - x_min)
            cur_left, cur_right = self.region.getRegion()
            cur_width = max(1e-12, float(cur_right - cur_left))
            old_center = 0.5 * (cur_left + cur_right)
            try:
                delta = float(event.delta())
            except Exception:
                try:
                    delta = float(event.angleDelta().y())
                except Exception:
                    delta = 0.0
            if abs(delta) < 1e-12:
                return super().eventFilter(obj, event)

            scale = 1.15 if delta > 0 else (1.0 / 1.15)
            target_width = cur_width * scale
            min_width = max(1e-6 * span, 1e-6)
            target_width = max(min_width, min(target_width, span))

            # 以鼠标所在时间点为锚点做缩放，保证“指哪缩哪”
            x_anchor = float(use_plot.vb.mapSceneToView(pos).x())
            x_anchor = max(x_min, min(x_anchor, x_max))
            if cur_width <= 1e-12:
                ratio = 0.5
            else:
                ratio = (x_anchor - cur_left) / cur_width
                ratio = max(0.0, min(1.0, ratio))

            left = x_anchor - ratio * target_width
            right = left + target_width
            if left < x_min:
                shift = x_min - left
                left += shift
                right += shift
            if right > x_max:
                shift = right - x_max
                left -= shift
                right -= shift
            left, right = self._clamp_x_window(left, right, x_min, x_max, prefer_width=target_width)
            new_center = 0.5 * (left + right)
            if abs(new_center - old_center) < 1e-12 and abs(target_width - cur_width) < 1e-12:
                try:
                    event.accept()
                except Exception:
                    pass
                return True
            self.region.setRegion((left, right))
            try:
                event.accept()
            except Exception:
                pass
            return True

        if et == QtCore.QEvent.GraphicsSceneMousePress:
            try:
                if event.button() != QtCore.Qt.LeftButton:
                    return super().eventFilter(obj, event)
                pos = event.scenePos()
            except Exception:
                return super().eventFilter(obj, event)

            active_plot = self._find_active_plot_by_scene_pos(pos)
            if active_plot is None:
                self._drag_select_candidate = False
                return super().eventFilter(obj, event)

            self._drag_select_candidate = True
            self._drag_select_active = False
            self._drag_start_scene_pos = QtCore.QPointF(pos)
            self._drag_active_plot = active_plot
            try:
                self._drag_start_x = float(active_plot.vb.mapSceneToView(pos).x())
            except Exception:
                self._drag_start_x = None
            self._clear_drag_preview()
            return super().eventFilter(obj, event)

        if et == QtCore.QEvent.GraphicsSceneMouseMove:
            if not self._drag_select_candidate or self._drag_active_plot is None or self._drag_start_scene_pos is None:
                return super().eventFilter(obj, event)
            try:
                if not (event.buttons() & QtCore.Qt.LeftButton):
                    return super().eventFilter(obj, event)
                pos = event.scenePos()
            except Exception:
                return super().eventFilter(obj, event)

            dx = abs(float(pos.x()) - float(self._drag_start_scene_pos.x()))
            if (not self._drag_select_active) and dx >= self._drag_threshold_px:
                self._drag_select_active = True
                self._drag_preview_region = pg.LinearRegionItem(
                    values=[self._drag_start_x, self._drag_start_x],
                    brush=pg.mkBrush(120, 120, 200, 50),
                    pen=pg.mkPen((70, 70, 140), width=1),
                    movable=False,
                )
                self._drag_preview_region.setZValue(30)
                self._drag_active_plot.addItem(self._drag_preview_region)

            if self._drag_select_active and self._drag_preview_region is not None and self._drag_start_x is not None:
                cur_x = float(self._drag_active_plot.vb.mapSceneToView(pos).x())
                left, right = (self._drag_start_x, cur_x) if self._drag_start_x <= cur_x else (cur_x, self._drag_start_x)
                if hasattr(self, "_x_domain_min") and hasattr(self, "_x_domain_max"):
                    x_min = float(self._x_domain_min)
                    x_max = float(self._x_domain_max)
                    left = max(x_min, min(left, x_max))
                    right = max(x_min, min(right, x_max))
                self._drag_preview_region.setRegion((left, right))
                try:
                    event.accept()
                except Exception:
                    pass
                return True
            return super().eventFilter(obj, event)

        if et == QtCore.QEvent.GraphicsSceneMouseRelease:
            if not self._drag_select_candidate:
                return super().eventFilter(obj, event)
            try:
                is_left_release = event.button() == QtCore.Qt.LeftButton
                pos = event.scenePos()
            except Exception:
                is_left_release = False
                pos = None

            if is_left_release and self._drag_select_active and self._drag_active_plot is not None and pos is not None and self._drag_start_x is not None:
                cur_x = float(self._drag_active_plot.vb.mapSceneToView(pos).x())
                left, right = (self._drag_start_x, cur_x) if self._drag_start_x <= cur_x else (cur_x, self._drag_start_x)
                if hasattr(self, "_x_domain_min") and hasattr(self, "_x_domain_max"):
                    x_min = float(self._x_domain_min)
                    x_max = float(self._x_domain_max)
                    left = max(x_min, min(left, x_max))
                    right = max(x_min, min(right, x_max))
                if abs(right - left) > 1e-12 and hasattr(self, "region"):
                    self.region.setRegion((left, right))
                    self._suppress_next_click = True

                self._clear_drag_preview()
                self._drag_select_candidate = False
                self._drag_select_active = False
                self._drag_start_scene_pos = None
                self._drag_start_x = None
                self._drag_active_plot = None
                try:
                    event.accept()
                except Exception:
                    pass
                return True

            self._clear_drag_preview()
            self._drag_select_candidate = False
            self._drag_select_active = False
            self._drag_start_scene_pos = None
            self._drag_start_x = None
            self._drag_active_plot = None
            return super().eventFilter(obj, event)

        return super().eventFilter(obj, event)

    def _format_time_text(self, x_actual: float) -> str:
        if self.is_time_axis:
            ts_ms = int(round(x_actual * 1000.0))
            return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        idx = int(np.searchsorted(self.x_display, x_actual))
        idx = max(0, min(idx, len(self.x_display) - 1))
        return f"样本: {idx}"

    def _build_value_lines(self, idx: int) -> List[str]:
        x_actual = self.x_display[idx]
        lines = []
        if self.is_time_axis:
            lines.append(f"时间: {self._format_time_text(x_actual)}")
        else:
            lines.append(f"样本: {idx}")

        for col in self.selected_cols:
            arr = self._get_numeric_column_array(col)
            val = arr[idx] if 0 <= idx < len(arr) else np.nan
            if pd.isna(val):
                sval = "NaN"
            else:
                try:
                    sval = f"{float(val):.6g}"
                except Exception:
                    sval = str(val)
            lines.append(f"{col}: {sval}")
        return lines

    def _compute_hover_gap_threshold(self) -> float:
        if self.x_display.size < 2:
            return 0.0
        diffs = np.diff(self.x_display.astype(np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 0.0
        if diffs.size < 5:
            return float(max(np.nanmax(diffs) * 2.0, 1e-9))
        med = float(np.nanmedian(diffs))
        p95 = float(np.nanpercentile(diffs, 95))
        return float(max(med * 5.0, p95 * 2.0, 1e-9))

    def _pick_hover_index(self, x_cursor: float) -> Optional[int]:
        if self.x_display.size == 0 or not np.isfinite(x_cursor):
            return None

        x = self.x_display
        n = len(x)
        threshold = float(self._hover_gap_threshold)
        idx = int(np.searchsorted(x, x_cursor, side="left"))

        if idx <= 0:
            if threshold <= 0.0:
                return 0
            return 0 if abs(float(x[0]) - x_cursor) <= threshold else None
        if idx >= n:
            if threshold <= 0.0:
                return n - 1
            return (n - 1) if abs(float(x[-1]) - x_cursor) <= threshold else None

        left_i = idx - 1
        right_i = idx
        left_x = float(x[left_i])
        right_x = float(x[right_i])
        gap = right_x - left_x
        if threshold > 0.0 and gap > threshold:
            return None

        chosen = left_i if abs(x_cursor - left_x) <= abs(right_x - x_cursor) else right_i
        if threshold > 0.0 and abs(float(x[chosen]) - x_cursor) > threshold:
            return None
        return chosen

    def _build_hover_value_lines(self, x_cursor: float, idx: Optional[int]) -> List[str]:
        lines = []
        if self.is_time_axis:
            ts_ms = int(round(x_cursor * 1000.0))
            t_text = pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            lines.append(f"时间: {t_text}")
        else:
            lines.append(f"样本位置: {x_cursor:.3f}")

        for col in self.selected_cols:
            sval = ""
            if idx is not None:
                arr = self._get_numeric_column_array(col)
                val = arr[idx] if 0 <= idx < len(arr) else np.nan
                if not pd.isna(val):
                    try:
                        sval = f"{float(val):.6g}"
                    except Exception:
                        sval = str(val)
            lines.append(f"{col}: {sval}")
        return lines

    def _update_current_value_panel_by_x(self, x_cursor: float) -> None:
        idx = self._pick_hover_index(x_cursor)
        for line in self.vlines:
            line.setPos(x_cursor)
        lines = self._build_hover_value_lines(x_cursor, idx)
        self.value_display.setPlainText("\n".join(lines))

    def _update_current_value_panel(self, idx: int) -> None:
        x_actual = self.x_display[idx]
        for line in self.vlines:
            line.setPos(x_actual)
        lines = self._build_value_lines(idx)
        self.value_display.setPlainText("\n".join(lines))

    def _freeze_marker_at_index(self, idx: int) -> None:
        slot = self.marker_next_slot
        x_actual = self.x_display[idx]
        self.marker_snapshots[slot] = {
            "x": float(x_actual),
            "time": self._format_time_text(x_actual),
            "text": "\n".join(self._build_value_lines(idx)),
        }
        self.marker_next_slot = (self.marker_next_slot + 1) % len(self.marker_snapshots)
        self._refresh_marker_views()
        self.statusBar().showMessage(f"已记录标记点{slot + 1}")

    def _refresh_marker_views(self) -> None:
        for i, snap in enumerate(self.marker_snapshots):
            if snap is None:
                self.marker_time_labels[i].setText("未标记")
                self.marker_text_edits[i].setPlainText("（空）")
                continue
            self.marker_time_labels[i].setText(snap["time"])
            self.marker_text_edits[i].setPlainText(snap["text"])
        self._refresh_marker_graphics()

    def _refresh_marker_graphics(self) -> None:
        if not self.plot_items:
            return

        for slot in range(3):
            snap = self.marker_snapshots[slot]
            lines = self.marker_slot_lines[slot] if slot < len(self.marker_slot_lines) else []
            if snap is None or "x" not in snap:
                for line in lines:
                    line.hide()
                continue
            x = float(snap["x"])
            for line in lines:
                line.setPos(x)
                line.show()

        if not self.plot_items:
            return
        bottom_plot = self.plot_items[-1]
        try:
            y_min, y_max = bottom_plot.viewRange()[1]
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                y_min, y_max = -1.0, 1.0
            x_min, x_max = bottom_plot.viewRange()[0]
            x_span = max(1e-9, float(x_max - x_min))
            y_span = max(1e-9, float(y_max - y_min))
        except Exception:
            return

        y_base = y_min + y_span * 0.02
        self.marker_delete_targets = []
        for slot, snap in enumerate(self.marker_snapshots):
            label_item = self.marker_bottom_label_items[slot] if slot < len(self.marker_bottom_label_items) else None
            if snap is None or "x" not in snap:
                if label_item is not None:
                    label_item.hide()
                continue
            x = float(snap["x"])
            if label_item is not None:
                label_item.setText(f"标记点{slot + 1}", color=self.marker_colors[slot])
                label_item.setPos(x, y_base)
                label_item.show()

    def _clear_marker(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.marker_snapshots):
            return
        self.marker_snapshots[idx] = None
        self._refresh_marker_views()
        self.statusBar().showMessage(f"已清除标记点{idx + 1}")

    def _clear_all_markers(self, silent: bool = False) -> None:
        self.marker_snapshots = [None, None, None]
        self.marker_next_slot = 0
        self._refresh_marker_views()
        if not silent:
            self.statusBar().showMessage("已清空全部标记")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSV多信号波形查看器")
    parser.add_argument("csv", nargs="?", default=None, help="CSV文件路径（可选，不传则启动后手动打开）")
    parser.add_argument("--time-col", default=None, help="指定时间列名（可选）")
    return parser


def configure_style() -> None:
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")


def set_chinese_font(app: QtWidgets.QApplication) -> None:
    preferred_fonts = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    db = QtGui.QFontDatabase()
    available = set(db.families())
    chosen = None
    for f in preferred_fonts:
        if f in available:
            chosen = f
            break

    if chosen:
        font = QtGui.QFont(chosen, 10)
        app.setFont(font)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    log_path = setup_runtime_diagnostics()
    install_exception_hooks()
    set_chinese_font(app)
    configure_style()

    init_df = pd.DataFrame({"示例信号": [0.0, 1.0, 0.0]})
    viewer = CsvWaveViewer(init_df, None)
    viewer.show()
    viewer.statusBar().showMessage(f"就绪 | 诊断日志: {log_path}")

    if args.csv is None:
        QtCore.QTimer.singleShot(0, viewer._open_csv_dialog)
    else:
        QtCore.QTimer.singleShot(0, lambda: viewer.start_load_csv(args.csv, args.time_col))

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

