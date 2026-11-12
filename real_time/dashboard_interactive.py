"""
dashboard_interactive.py
------------------------
Dashboard interativo para o pipeline EMG → SNN.

Painel lateral com checkboxes permite mostrar/ocultar qualquer visualização
durante a aquisição sem reiniciar o pipeline.

Dependências:
    pip install pyqtgraph PyQt5

Interface compatível com StreamingDashboard original:
    os métodos push(), start() e o atributo running funcionam da mesma forma.
"""

import sys
import numpy as np
import queue
from collections import deque

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import torch

# ── Constantes visuais ────────────────────────────────────────────────────────
CLASS_NAMES      = ["Rest", "Diameter Grasp"]
CLASS_COLORS_HEX = ["#4CAF50", "#2196F3", "#FF9800", "#F52834"]
CH_COLORS = [
    "#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b",
    "#cc5de8", "#ff922b", "#74c0fc", "#f783ac",
]
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
TEXT_DIM   = "#8b949e"
TEXT_BRIGHT = "#f0f6fc"
ACCENT     = "#58a6ff"
BORDER     = "#30363d"


class InteractiveDashboard(QtWidgets.QMainWindow):
    """
    Parâmetros
    ----------
    n_channels      : número de canais EMG (um plot por canal)
    n_classes       : número de classes de movimento
    n_steps         : passos de tempo do SNN (eixo x do raster)
    history_len     : janelas no histórico de classificação
    update_ms       : intervalo de atualização dos plots em ms
    display_samples : amostras visíveis por canal no plot de EMG
    """

    def __init__(
        self,
        n_channels: int     = 8,
        n_classes: int      = 4,
        n_steps: int        = 25,
        history_len: int    = 200,
        update_ms: int      = 100,
        display_samples: int = 1000,
    ):
        self._app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication(sys.argv)
        )
        pg.setConfigOption("background", DARK_BG)
        pg.setConfigOption("foreground", TEXT_DIM)
        pg.setConfigOption("antialias",  True)

        super().__init__()

        self.n_channels      = n_channels
        self.n_classes       = n_classes
        self.n_steps         = n_steps
        self.history_len     = history_len
        self.update_ms       = update_ms
        self.display_samples = display_samples

        self.data_queue = queue.Queue(maxsize=60)
        self.running    = True

        # Buffers de dados
        self._emg_bufs     = [deque(maxlen=display_samples) for _ in range(n_channels)]
        self._class_hist   = deque(maxlen=history_len)
        self._time_hist    = deque(maxlen=history_len)
        self._label_hist   = deque(maxlen=history_len)

        self._build_ui()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._drain_queue)

    # ─────────────────────────────────────────────────────────────────────────
    # Construção da interface
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("EMG → SNN | Monitor Interativo")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_BRIGHT};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_plot_area(), stretch=1)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setFixedWidth(210)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"background-color: {PANEL_BG}; border: none;"
            f" border-right: 1px solid {BORDER};"
        )

        content = QtWidgets.QWidget()
        content.setStyleSheet(f"background-color: {PANEL_BG};")
        vbox = QtWidgets.QVBoxLayout(content)
        vbox.setContentsMargins(10, 14, 10, 14)
        vbox.setSpacing(8)

        hdr = QtWidgets.QLabel("PAINÉIS")
        hdr.setStyleSheet(
            f"color: {ACCENT}; font-weight: bold; font-size: 11px; letter-spacing: 1px;"
        )
        vbox.addWidget(hdr)

        # ── Grupo EMG ─────────────────────────────────────────────────────────
        emg_box = self._group("EMG — canais")
        emg_vbox = QtWidgets.QVBoxLayout()
        emg_vbox.setSpacing(2)

        # Botões "Todos / Nenhum"
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_all  = self._small_btn("Todos")
        self._btn_none = self._small_btn("Nenhum")
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        emg_vbox.addLayout(btn_row)

        self._cb_emg: list[QtWidgets.QCheckBox] = []
        for ch in range(self.n_channels):
            cb = QtWidgets.QCheckBox(f"Canal {ch + 1}")
            cb.setChecked(True)
            cb.setStyleSheet(
                f"color: {CH_COLORS[ch % len(CH_COLORS)]}; font-size: 11px;"
            )
            cb.stateChanged.connect(
                lambda s, i=ch: self._emg_plots[i].setVisible(bool(s))
            )
            self._cb_emg.append(cb)
            emg_vbox.addWidget(cb)

        self._btn_all.clicked.connect(
            lambda: [cb.setChecked(True) for cb in self._cb_emg]
        )
        self._btn_none.clicked.connect(
            lambda: [cb.setChecked(False) for cb in self._cb_emg]
        )

        emg_box.setLayout(emg_vbox)
        vbox.addWidget(emg_box)

        # ── Grupo Classificação ───────────────────────────────────────────────
        clf_box = self._group("Classificação")
        clf_vbox = QtWidgets.QVBoxLayout()
        clf_vbox.setSpacing(2)
        self._cb_class   = self._cb_item("Classe predita", True,  clf_vbox)
        self._cb_history = self._cb_item("Histórico",      True,  clf_vbox)
        self._cb_labels  = self._cb_item("True Labels",    False, clf_vbox)
        clf_box.setLayout(clf_vbox)
        vbox.addWidget(clf_box)

        # ── Grupo SNN ─────────────────────────────────────────────────────────
        snn_box = self._group("SNN internals")
        snn_vbox = QtWidgets.QVBoxLayout()
        snn_vbox.setSpacing(2)
        self._cb_raster = self._cb_item("Delta spike raster", True,  snn_vbox)
        self._cb_spikes = self._cb_item("Contagem de spikes", True,  snn_vbox)
        snn_box.setLayout(snn_vbox)
        vbox.addWidget(snn_box)

        vbox.addStretch()
        scroll.setWidget(content)
        return scroll

    # ── Área de plots ─────────────────────────────────────────────────────────

    def _build_plot_area(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background-color: {DARK_BG}; border: none;")

        container = QtWidgets.QWidget()
        container.setStyleSheet(f"background-color: {DARK_BG};")
        self._plot_vbox = QtWidgets.QVBoxLayout(container)
        self._plot_vbox.setContentsMargins(8, 8, 8, 8)
        self._plot_vbox.setSpacing(6)

        # ── EMG (um widget por canal) ─────────────────────────────────────────
        self._emg_plots: list[pg.PlotWidget]    = []
        self._emg_curves: list[pg.PlotDataItem] = []

        for ch in range(self.n_channels):
            color = CH_COLORS[ch % len(CH_COLORS)]
            pw = pg.PlotWidget(
                title=f"<span style='color:{color}'>Canal {ch + 1}</span>"
            )
            pw.setFixedHeight(110)
            pw.setBackground(PANEL_BG)
            pw.showGrid(x=False, y=True, alpha=0.15)
            pw.getAxis("bottom").setStyle(showValues=False)
            pw.getAxis("left").setStyle(
                tickFont=QtGui.QFont("monospace", 7), tickLength=3
            )
            curve = pw.plot(pen=pg.mkPen(color=color, width=1))
            self._emg_plots.append(pw)
            self._emg_curves.append(curve)
            self._plot_vbox.addWidget(pw)

        # ── Classe predita ────────────────────────────────────────────────────
        self._class_frame = self._build_class_frame()
        self._plot_vbox.addWidget(self._class_frame)

        # ── Spike raster ──────────────────────────────────────────────────────
        self._raster_plot = self._make_pw("Delta Spikes  (↑ ciano = up  /  ↓ salmão = down)", 170)
        self._plot_vbox.addWidget(self._raster_plot)

        # ── Contagem de spikes (barras verticais) ─────────────────────────────
        self._spikes_plot = self._make_pw("Contagem de Spikes por Classe", 140)
        x_ticks = [
            (i, CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i))
            for i in range(self.n_classes)
        ]
        self._spikes_plot.getAxis("bottom").setTicks([x_ticks])
        self._spike_bars = pg.BarGraphItem(
            x=np.arange(self.n_classes),
            height=np.zeros(self.n_classes),
            width=0.6,
            brushes=[
                pg.mkBrush(CLASS_COLORS_HEX[i % len(CLASS_COLORS_HEX)])
                for i in range(self.n_classes)
            ],
        )
        self._spikes_plot.addItem(self._spike_bars)
        self._plot_vbox.addWidget(self._spikes_plot)

        # ── Histórico de classificação ────────────────────────────────────────
        self._history_plot = self._make_pw("Histórico de Classificação", 160)
        y_ticks = [
            (i, CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i))
            for i in range(self.n_classes)
        ]
        self._history_plot.getAxis("left").setTicks([y_ticks])
        self._history_plot.setYRange(-0.5, self.n_classes - 0.5)
        self._history_scatter = pg.ScatterPlotItem(size=7)
        self._history_plot.addItem(self._history_scatter)
        self._plot_vbox.addWidget(self._history_plot)

        # ── True labels ───────────────────────────────────────────────────────
        self._labels_plot = self._make_pw("True Labels ao Longo do Tempo", 130)
        self._labels_scatter = pg.ScatterPlotItem(
            size=5, brush=pg.mkBrush(TEXT_DIM)
        )
        self._labels_plot.addItem(self._labels_scatter)
        self._plot_vbox.addWidget(self._labels_plot)

        self._plot_vbox.addStretch()
        scroll.setWidget(container)

        # ── Conecta checkboxes → visibilidade ─────────────────────────────────
        self._cb_class.stateChanged.connect(
            lambda s: self._class_frame.setVisible(bool(s))
        )
        self._cb_raster.stateChanged.connect(
            lambda s: self._raster_plot.setVisible(bool(s))
        )
        self._cb_spikes.stateChanged.connect(
            lambda s: self._spikes_plot.setVisible(bool(s))
        )
        self._cb_history.stateChanged.connect(
            lambda s: self._history_plot.setVisible(bool(s))
        )
        self._cb_labels.stateChanged.connect(
            lambda s: self._labels_plot.setVisible(bool(s))
        )

        # Estado inicial: True Labels oculto
        self._labels_plot.setVisible(False)

        return scroll

    def _build_class_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setFixedHeight(90)
        frame.setStyleSheet(
            f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;"
        )
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(20, 8, 20, 8)
        row.setSpacing(16)

        self._lbl_class = QtWidgets.QLabel("---")
        self._lbl_class.setStyleSheet(
            f"color: {CLASS_COLORS_HEX[0]}; font-size: 30px; font-weight: bold;"
            f" background: transparent; border: none;"
        )
        self._lbl_class.setAlignment(QtCore.Qt.AlignCenter)

        meta = QtWidgets.QVBoxLayout()
        self._lbl_time = QtWidgets.QLabel("t = 0.00 s")
        self._lbl_true = QtWidgets.QLabel("Label: —")
        for lbl in (self._lbl_time, self._lbl_true):
            lbl.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 10px; background: transparent; border: none;"
            )
            meta.addWidget(lbl)

        meta_w = QtWidgets.QWidget()
        meta_w.setStyleSheet("background: transparent; border: none;")
        meta_w.setLayout(meta)

        row.addWidget(self._lbl_class, stretch=3)
        row.addWidget(meta_w,          stretch=1)
        return frame

    # ── Helpers de estilo ─────────────────────────────────────────────────────

    def _group(self, title: str) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox(title)
        g.setStyleSheet(
            f"QGroupBox {{ color: {TEXT_DIM}; font-size: 10px; font-weight: bold;"
            f"  border: 1px solid {BORDER}; border-radius: 4px;"
            f"  margin-top: 8px; padding-top: 4px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 2px; }}"
        )
        return g

    def _cb_item(
        self, label: str, checked: bool, layout: QtWidgets.QVBoxLayout
    ) -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(checked)
        cb.setStyleSheet(f"color: {TEXT_BRIGHT}; font-size: 11px;")
        layout.addWidget(cb)
        return cb

    def _small_btn(self, label: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(label)
        btn.setFixedHeight(22)
        btn.setStyleSheet(
            f"QPushButton {{ background: #21262d; color: {TEXT_DIM};"
            f"  border: 1px solid {BORDER}; border-radius: 3px;"
            f"  padding: 2px 6px; font-size: 10px; }}"
            f"QPushButton:hover {{ background: {BORDER}; }}"
        )
        return btn

    def _make_pw(self, title: str, height: int) -> pg.PlotWidget:
        pw = pg.PlotWidget(
            title=f"<span style='color:{ACCENT}'>{title}</span>"
        )
        pw.setFixedHeight(height)
        pw.setBackground(PANEL_BG)
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.getAxis("left").setStyle(
            tickFont=QtGui.QFont("monospace", 7), tickLength=3
        )
        pw.getAxis("bottom").setStyle(
            tickFont=QtGui.QFont("monospace", 7), tickLength=3
        )
        return pw

    # ─────────────────────────────────────────────────────────────────────────
    # API pública (compatível com StreamingDashboard)
    # ─────────────────────────────────────────────────────────────────────────

    def push(self, emg_window, pred_class, spike_count, spike_rec, timestamp, true_label):
        """Enfileira dados vindos da thread do pipeline. Thread-safe."""
        try:
            self.data_queue.put_nowait(
                (emg_window, pred_class, spike_count, spike_rec, timestamp, true_label)
            )
        except queue.Full:
            pass  # descarta se o plot não acompanhar

    def start(self, pipeline_fn):
        """Inicia a thread do pipeline e o loop Qt (bloqueante até fechar a janela)."""
        import threading
        t = threading.Thread(target=pipeline_fn, daemon=True)
        t.start()
        self._timer.start(self.update_ms)
        self.show()
        self._app.exec_()
        self.running = False

    def closeEvent(self, event):
        self.running = False
        self._timer.stop()
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Loop de atualização
    # ─────────────────────────────────────────────────────────────────────────

    def _drain_queue(self):
        """Drena toda a fila de uma vez e faz um único redesenho por ciclo."""
        items = []
        while True:
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        if not items:
            return

        # Acumula EMG de todos os itens no buffer antes de desenhar
        for emg_w, *_ in items:
            for ch in range(min(self.n_channels, emg_w.shape[1])):
                self._emg_bufs[ch].extend(emg_w[:, ch].tolist())

        # Para painéis de "snapshot" usa sempre o item mais recente
        _, pred_class, spike_count, spike_rec, timestamp, true_label = items[-1]

        # ── EMG ───────────────────────────────────────────────────────────────
        for ch in range(self.n_channels):
            if self._emg_plots[ch].isVisible():
                data = np.array(self._emg_bufs[ch])
                if data.size:
                    self._emg_curves[ch].setData(data)

        # ── Classe predita ────────────────────────────────────────────────────
        if self._class_frame.isVisible():
            idx   = pred_class % len(CLASS_NAMES)
            name  = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(pred_class)
            color = CLASS_COLORS_HEX[pred_class % len(CLASS_COLORS_HEX)]
            self._lbl_class.setText(name)
            self._lbl_class.setStyleSheet(
                f"color: {color}; font-size: 30px; font-weight: bold;"
                f" background: transparent; border: none;"
            )
            self._lbl_time.setText(f"t = {timestamp:.2f} s")
            if true_label is not None:
                self._lbl_true.setText(f"Label: {true_label}")

        # ── Spike raster ──────────────────────────────────────────────────────
        if self._raster_plot.isVisible():
            raster = (
                spike_rec.numpy()
                if isinstance(spike_rec, torch.Tensor)
                else np.array(spike_rec)
            )
            self._raster_plot.clear()
            n_ch = raster.shape[1] // 2
            for ch in range(n_ch):
                ups   = np.where(raster[:, ch] > 0.5)[0]
                downs = np.where(raster[:, ch + n_ch] > 0.5)[0]
                if len(ups):
                    self._raster_plot.plot(
                        ups, [ch * 2 + 1] * len(ups),
                        pen=None, symbol="|", symbolSize=10,
                        symbolBrush=pg.mkBrush("cyan"), symbolPen=None,
                    )
                if len(downs):
                    self._raster_plot.plot(
                        downs, [ch * 2] * len(downs),
                        pen=None, symbol="|", symbolSize=10,
                        symbolBrush=pg.mkBrush("salmon"), symbolPen=None,
                    )

        # ── Contagem de spikes ────────────────────────────────────────────────
        if self._spikes_plot.isVisible():
            counts = (
                spike_count[0].numpy()
                if isinstance(spike_count[0], torch.Tensor)
                else np.array(spike_count[0])
            )
            self._spike_bars.setOpts(height=counts.tolist())
            if counts.max() > 0:
                self._spikes_plot.setYRange(0, counts.max() * 1.1)

        # ── Histórico ─────────────────────────────────────────────────────────
        self._class_hist.append(pred_class)
        self._time_hist.append(timestamp)

        if self._history_plot.isVisible():
            times   = np.array(self._time_hist)
            classes = np.array(self._class_hist)
            brushes = [
                pg.mkBrush(CLASS_COLORS_HEX[int(c) % len(CLASS_COLORS_HEX)])
                for c in classes
            ]
            self._history_scatter.setData(x=times, y=classes.astype(float), brush=brushes)
            if len(times) > 1:
                self._history_plot.setXRange(times[0], max(times[-1], times[0] + 1))

        # ── True Labels ───────────────────────────────────────────────────────
        self._label_hist.append(true_label)

        if self._labels_plot.isVisible():
            times  = np.array(self._time_hist)
            labels = np.array(
                [l if l is not None else np.nan for l in self._label_hist],
                dtype=float,
            )
            valid = ~np.isnan(labels)
            if valid.any():
                self._labels_scatter.setData(x=times[valid], y=labels[valid])
                self._labels_plot.setXRange(times[0], max(times[-1], times[0] + 1))
