import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


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
    finished = QtCore.Signal(object, object, str)
    error = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(self, path: str, forced_time_col: Optional[str] = None):
        super().__init__()
        self.path = path
        self.forced_time_col = forced_time_col
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

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

                self.progress.emit(100, "加载完成")
                self.finished.emit(df, time_col, os.path.basename(self.path))
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

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.left_panel, 0)
        layout.addWidget(self.plot_widget, 1)
        layout.addWidget(self.value_panel, 0)
        self.setCentralWidget(central)

        self.plot_items = []
        self.plot_cols: List[str] = []
        self.curves = {}
        self.vlines = []
        self.hover_proxy = None
        self.selected_cols: List[str] = []

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

        tips = QtWidgets.QLabel(
            "使用说明:\n"
            "1) 在左侧搜索并勾选信号（默认不勾选）\n"
            "2) 勾选一个就会在右侧新增一个波形\n"
            "3) 底部总览图拖拽灰色区域选择时间段\n"
            "4) 鼠标移动到任意子图，右侧显示当前时刻各信号值"
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
        panel.setMaximumWidth(360)

        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(6)

        title = QtWidgets.QLabel("当前光标数值")
        title.setStyleSheet("font-weight: bold;")
        vbox.addWidget(title)

        self.value_display = QtWidgets.QPlainTextEdit()
        self.value_display.setReadOnly(True)
        self.value_display.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.value_display.setPlainText("鼠标移动到波形区域后，这里会显示当前时刻各信号值。")
        vbox.addWidget(self.value_display, 1)
        return panel

    def _make_axis_items(self) -> Optional[dict]:
        if not self.is_time_axis:
            return None
        # 强制按 UTC 基准显示，避免系统时区导致 +8/-8 小时偏移
        return {"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)}

    def _bind_left_signals(self) -> None:
        self.signal_list.itemChanged.connect(self._on_signal_item_changed)
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.search_edit.textChanged.connect(self._filter_signals)

    def _set_data(self, df: pd.DataFrame, time_col: Optional[str]) -> None:
        self.df = df
        self.time_col = time_col
        self.x_raw, self.x_display, self.is_time_axis = self._build_x_axis(df, time_col)

        self.numeric_cols = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not self.numeric_cols:
            raise ValueError("CSV中没有可绘制的数值列。")

        new_left = self._build_left_panel()
        old_left = self.left_panel
        parent_layout = self.centralWidget().layout()
        parent_layout.replaceWidget(old_left, new_left)
        old_left.deleteLater()
        self.left_panel = new_left

        self._bind_left_signals()
        self._update_selected_count()
        self._init_plots()

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

    def _on_load_finished(self, df: pd.DataFrame, time_col: Optional[str], file_name: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()

        try:
            self._set_data(df, time_col)
            self.setWindowTitle(f"CSV多信号波形查看器 - {file_name}")
            if time_col is None:
                self.statusBar().showMessage(f"加载完成: {file_name}（未识别到时间列，当前为样本轴）")
            else:
                self.statusBar().showMessage(f"加载完成: {file_name}（时间列: {time_col}）")
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

    def _init_plots(self) -> None:
        self.plot_widget.clear()
        self.plot_items.clear()
        self.plot_cols.clear()
        self.curves.clear()
        self.vlines.clear()

        checked_cols = self._get_checked_columns()
        self.selected_cols = checked_cols

        if not checked_cols:
            label = self.plot_widget.addLabel("请在左侧勾选信号后显示波形", row=0, col=0)
            label.setText("请在左侧勾选信号后显示波形")
            self.value_display.setPlainText("请先勾选信号。")
            return

        for i, col in enumerate(checked_cols):
            axis_items = self._make_axis_items()
            if axis_items is None:
                p = self.plot_widget.addPlot(row=i, col=0, title=str(col))
            else:
                p = self.plot_widget.addPlot(row=i, col=0, title=str(col), axisItems=axis_items)

            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", str(col))

            y = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=np.float64)
            curve = p.plot(
                self.x_display,
                y,
                pen=pg.mkPen(color=pg.intColor(i, hues=max(8, len(checked_cols))), width=1),
                autoDownsample=True,
                downsampleMethod="peak",
                clipToView=True,
            )
            self.curves[col] = curve

            if i > 0:
                p.setXLink(self.plot_items[0])
            else:
                p.setLabel("bottom", "时间" if self.is_time_axis else "样本点")

            vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 80, 80), width=1))
            p.addItem(vline, ignoreBounds=True)
            self.vlines.append(vline)
            self.plot_items.append(p)
            self.plot_cols.append(col)

        overview_row = len(checked_cols)
        axis_items = self._make_axis_items()
        if axis_items is None:
            self.overview_plot = self.plot_widget.addPlot(row=overview_row, col=0, title="时间段选择")
        else:
            self.overview_plot = self.plot_widget.addPlot(
                row=overview_row, col=0, title="时间段选择", axisItems=axis_items
            )

        self.overview_plot.showGrid(x=True, y=True, alpha=0.2)
        self.overview_plot.setMaximumHeight(180)

        first_col = checked_cols[0]
        y_overview = pd.to_numeric(self.df[first_col], errors="coerce").to_numpy(dtype=np.float64)
        self.overview_plot.plot(self.x_display, y_overview, pen=pg.mkPen((120, 120, 120), width=1))

        x_min = float(np.nanmin(self.x_display))
        x_max = float(np.nanmax(self.x_display))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            x_min, x_max = 0.0, max(1.0, float(len(self.x_display) - 1))

        initial_left = x_min + (x_max - x_min) * 0.15
        initial_right = x_min + (x_max - x_min) * 0.65
        self.region = pg.LinearRegionItem(
            values=[initial_left, initial_right],
            brush=pg.mkBrush(100, 100, 180, 45),
            pen=pg.mkPen((70, 70, 140), width=1),
        )
        self.region.setZValue(10)
        self.overview_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self._on_region_changed()

        if self.hover_proxy is not None:
            try:
                self.hover_proxy.disconnect()
            except Exception:
                pass
        self.hover_proxy = pg.SignalProxy(self.plot_items[0].scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
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
            y = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=np.float64)
            seg = y[i0:i1]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            if seg.size >= 30:
                lo, hi = np.nanpercentile(seg, [1, 99])
            else:
                lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))

            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if lo == hi:
                pad = max(1.0, abs(lo) * 0.1)
                p.setYRange(lo - pad, hi + pad, padding=0)
            else:
                pad = (hi - lo) * 0.08
                p.setYRange(lo - pad, hi + pad, padding=0)

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

    def _on_region_changed(self) -> None:
        if not self.plot_items:
            return
        left, right = self.region.getRegion()
        for p in self.plot_items:
            p.setXRange(left, right, padding=0)
        self._update_all_y_ranges()

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

        idx = int(np.searchsorted(self.x_display, x_val))
        idx = max(0, min(idx, len(self.x_display) - 1))

        x_actual = self.x_display[idx]
        for line in self.vlines:
            line.setPos(x_actual)

        lines = []
        if self.is_time_axis:
            ts_ms = int(round(x_actual * 1000.0))
            ts_text = pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            lines.append(f"时间: {ts_text}")
        else:
            lines.append(f"样本: {idx}")

        for col in self.selected_cols:
            val = self.df[col].iloc[idx]
            if pd.isna(val):
                sval = "NaN"
            else:
                try:
                    sval = f"{float(val):.6g}"
                except Exception:
                    sval = str(val)
            lines.append(f"{col}: {sval}")

        self.value_display.setPlainText("\n".join(lines))


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
    set_chinese_font(app)
    configure_style()

    init_df = pd.DataFrame({"示例信号": [0.0, 1.0, 0.0]})
    viewer = CsvWaveViewer(init_df, None)
    viewer.show()

    if args.csv is None:
        QtCore.QTimer.singleShot(0, viewer._open_csv_dialog)
    else:
        QtCore.QTimer.singleShot(0, lambda: viewer.start_load_csv(args.csv, args.time_col))

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
