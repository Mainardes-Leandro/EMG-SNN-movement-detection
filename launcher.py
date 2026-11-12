"""
launcher.py
-----------
Interface gráfica de configuração para o pipeline EMG → SNN.

Substitui os argumentos de linha de comando por uma janela com abas.
O real_time_run.py não precisa ser modificado — o launcher monta um
objeto de argumentos compatível e chama main() diretamente.

Uso:
    python launcher.py
"""

import sys
import os
import threading
import traceback
import types

from PyQt5 import QtWidgets, QtCore, QtGui

# ── Paleta ────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
INPUT_BG   = "#21262d"
BORDER     = "#30363d"
ACCENT     = "#58a6ff"
TEXT       = "#f0f6fc"
TEXT_DIM   = "#8b949e"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#d29922"

_BASE_STYLE = f"""
    QWidget       {{ background-color: {DARK_BG}; color: {TEXT}; font-size: 12px; }}
    QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL_BG}; }}
    QTabBar::tab  {{ background: {INPUT_BG}; color: {TEXT_DIM}; padding: 6px 18px;
                     border: 1px solid {BORDER}; border-bottom: none; border-radius: 4px 4px 0 0; }}
    QTabBar::tab:selected {{ background: {PANEL_BG}; color: {TEXT}; }}
    QGroupBox     {{ color: {TEXT_DIM}; font-size: 11px; font-weight: bold;
                     border: 1px solid {BORDER}; border-radius: 4px;
                     margin-top: 10px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QLineEdit, QComboBox, QSpinBox
                  {{ background: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
                     border-radius: 3px; padding: 3px 6px; }}
    QComboBox::drop-down {{ border: none; }}
    QCheckBox     {{ color: {TEXT}; spacing: 6px; }}
    QRadioButton  {{ color: {TEXT}; spacing: 6px; }}
    QLabel        {{ color: {TEXT_DIM}; }}
    QPushButton   {{ background: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
                     border-radius: 4px; padding: 5px 14px; }}
    QPushButton:hover   {{ background: {BORDER}; }}
    QPushButton:pressed {{ background: {ACCENT}; color: {DARK_BG}; }}
    QScrollBar:vertical {{ background: {INPUT_BG}; width: 8px; border-radius: 4px; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; }}
    QTextEdit     {{ background: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
                     border-radius: 3px; font-family: monospace; font-size: 11px; }}
"""


# ── Redirecionador de stdout/stderr → QTextEdit ───────────────────────────────

class _LogRedirect(QtCore.QObject):
    """Captura print()s da thread do pipeline e os emite como sinal Qt."""
    new_line = QtCore.pyqtSignal(str, str)   # (texto, nível: info/warn/error)

    def __init__(self, level="info"):
        super().__init__()
        self.level   = level
        self._orig   = sys.stdout if level == "info" else sys.stderr
        self._buffer = ""

    def write(self, text):
        self._orig.write(text)   # continua imprimindo no terminal também
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.new_line.emit(line, self.level)

    def flush(self):
        self._orig.flush()


# ── Utilitários de layout ─────────────────────────────────────────────────────

def _lbl(text):
    l = QtWidgets.QLabel(text)
    l.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
    return l


def _field(placeholder="", width=None):
    e = QtWidgets.QLineEdit()
    e.setPlaceholderText(placeholder)
    if width:
        e.setFixedWidth(width)
    return e


def _browse_btn():
    b = QtWidgets.QPushButton("Procurar…")
    b.setFixedWidth(90)
    return b


def _spin(min_val, max_val, default, step=1, width=90):
    s = QtWidgets.QSpinBox()
    s.setRange(min_val, max_val)
    s.setValue(default)
    s.setSingleStep(step)
    s.setFixedWidth(width)
    return s


def _combo(items, current=None, width=None):
    c = QtWidgets.QComboBox()
    c.addItems(items)
    if current and current in items:
        c.setCurrentText(current)
    if width:
        c.setFixedWidth(width)
    return c


def _row(*widgets, spacing=8):
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(spacing)
    for ww in widgets:
        if ww == "stretch":
            h.addStretch()
        else:
            h.addWidget(ww)
    return w


def _accent_btn(text, color=ACCENT):
    b = QtWidgets.QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background: {color}; color: {DARK_BG}; font-weight: bold;"
        f"  border: none; border-radius: 4px; padding: 7px 24px; font-size: 13px; }}"
        f"QPushButton:hover {{ background: #388bfd; }}"
        f"QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM}; }}"
    )
    return b


# ── Janela principal ──────────────────────────────────────────────────────────

class LauncherWindow(QtWidgets.QMainWindow):

    def __init__(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self._app = app
        super().__init__()

        self.setWindowTitle("EMG → SNN Pipeline — Configuração")
        self.setMinimumSize(680, 740)
        self.setStyleSheet(_BASE_STYLE)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Cabeçalho
        hdr = QtWidgets.QLabel("EMG → SNN Pipeline")
        hdr.setStyleSheet(
            f"color: {ACCENT}; font-size: 20px; font-weight: bold;"
        )
        root.addWidget(hdr)

        sub = QtWidgets.QLabel("Configure os parâmetros e clique em Iniciar.")
        sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        root.addWidget(sub)

        # Abas de configuração
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._tab_source(),    "  Fonte de dados  ")
        self._tabs.addTab(self._tab_model(),     "  Modelo SNN  ")
        self._tabs.addTab(self._tab_dashboard(), "  Dashboard  ")
        root.addWidget(self._tabs)

        # Log
        log_lbl = QtWidgets.QLabel("LOG")
        log_lbl.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        root.addWidget(log_lbl)

        self._log = QtWidgets.QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(180)
        root.addWidget(self._log)

        # Botões inferiores
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_start  = _accent_btn("▶  Iniciar")
        self._btn_cancel = QtWidgets.QPushButton("Fechar")
        btn_row.addStretch()
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_start)
        root.addLayout(btn_row)

        # Conexões
        self._btn_start.clicked.connect(self._start_pipeline)
        self._btn_cancel.clicked.connect(self.close)

        # Redireciona stdout/stderr para o log
        self._log_out = _LogRedirect("info")
        self._log_err = _LogRedirect("error")
        self._log_out.new_line.connect(self._append_log)
        self._log_err.new_line.connect(self._append_log)
        sys.stdout = self._log_out
        sys.stderr = self._log_err

        self._pipeline_thread = None

    # ── Aba: Fonte de dados ───────────────────────────────────────────────────

    def _tab_source(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(12, 12, 12, 12)

        # Seleção do modo
        mode_box = QtWidgets.QGroupBox("Modo de aquisição")
        mv = QtWidgets.QVBoxLayout(mode_box)
        self._rb_live  = QtWidgets.QRadioButton("Ao vivo — hardware ESP32/ADS1299")
        self._rb_mat   = QtWidgets.QRadioButton("Arquivo .mat (NinaPro ou similar)")
        self._rb_synth = QtWidgets.QRadioButton("Sintético (teste sem hardware)")
        self._rb_live.setChecked(True)
        for rb in (self._rb_live, self._rb_mat, self._rb_synth):
            rb.setStyleSheet(f"color: {TEXT}; spacing: 6px;")
            mv.addWidget(rb)
        v.addWidget(mode_box)

        # ── Opções ao vivo ────────────────────────────────────────────────────
        self._grp_live = QtWidgets.QGroupBox("Configuração do hardware")
        gl = QtWidgets.QFormLayout(self._grp_live)
        gl.setSpacing(8)

        self._cb_fs_hw = _combo(
            ["250", "500", "1000", "2000", "4000", "8000", "16000"], "1000", 110
        )
        self._cb_gain = _combo(["1","2","4","6","8","12","24"], "24", 80)
        self._cb_srb1 = QtWidgets.QCheckBox("Habilitar SRB1 (montagem referencial)")
        self._cb_srb1.setChecked(True)
        self._spin_devices = _combo(["Apenas device #1", "Devices #1 e #2"], width=180)

        gl.addRow(_lbl("Taxa do hardware (SPS):"), self._cb_fs_hw)
        gl.addRow(_lbl("Ganho dos canais:"),        self._cb_gain)
        gl.addRow(_lbl("Devices ativos:"),          self._spin_devices)
        gl.addRow("", self._cb_srb1)
        v.addWidget(self._grp_live)

        # ── Opções .mat ───────────────────────────────────────────────────────
        self._grp_mat = QtWidgets.QGroupBox("Arquivo .mat")
        gm = QtWidgets.QFormLayout(self._grp_mat)
        gm.setSpacing(8)

        self._mat_path   = _field("Caminho para o arquivo .mat")
        self._mat_signal = _field("emg", 120)
        self._mat_label  = _field("stimulus", 120)
        mat_browse       = _browse_btn()
        mat_browse.clicked.connect(
            lambda: self._browse_file(self._mat_path, "Arquivos MAT (*.mat)")
        )
        gm.addRow(_lbl("Arquivo:"),    _row(self._mat_path, mat_browse))
        gm.addRow(_lbl("Chave sinal:"), self._mat_signal)
        gm.addRow(_lbl("Chave label:"), self._mat_label)
        self._grp_mat.setVisible(False)
        v.addWidget(self._grp_mat)

        v.addStretch()

        # Visibilidade condicional
        self._rb_live.toggled.connect(
            lambda on: (self._grp_live.setVisible(on), self._grp_mat.setVisible(False))
        )
        self._rb_mat.toggled.connect(
            lambda on: (self._grp_mat.setVisible(on), self._grp_live.setVisible(False))
        )
        self._rb_synth.toggled.connect(
            lambda on: (self._grp_live.setVisible(not on and self._rb_live.isChecked()),
                        self._grp_mat.setVisible(False))
        )

        return w

    # ── Aba: Modelo SNN ──────────────────────────────────────────────────────

    def _tab_model(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(12, 12, 12, 12)

        # Pesos
        wgt_box = QtWidgets.QGroupBox("Pesos do modelo")
        wl = QtWidgets.QFormLayout(wgt_box)
        wl.setSpacing(8)
        self._weights_path = _field("Deixe vazio para pesos aleatórios (teste)")
        wgt_browse = _browse_btn()
        wgt_browse.clicked.connect(
            lambda: self._browse_file(
                self._weights_path, "Pesos PyTorch (*.pt *.pth)"
            )
        )
        wl.addRow(_lbl("Arquivo .pth:"), _row(self._weights_path, wgt_browse))
        v.addWidget(wgt_box)

        # Parâmetros
        param_box = QtWidgets.QGroupBox("Parâmetros da arquitetura")
        pl = QtWidgets.QFormLayout(param_box)
        pl.setSpacing(8)
        self._spin_channels = _spin(1, 64,  8)
        self._spin_classes  = _spin(2, 20,  2)
        self._spin_steps    = _spin(1, 100, 10)
        self._spin_fs       = _spin(1, 2000, 200)
        pl.addRow(_lbl("Canais EMG:"),       self._spin_channels)
        pl.addRow(_lbl("Classes:"),          self._spin_classes)
        pl.addRow(_lbl("Passos de tempo:"),  self._spin_steps)
        pl.addRow(_lbl("Fs do modelo (Hz):"), self._spin_fs)
        v.addWidget(param_box)

        # Janelamento
        win_box = QtWidgets.QGroupBox("Janelamento")
        wn = QtWidgets.QFormLayout(win_box)
        wn.setSpacing(8)
        self._spin_window  = _spin(100, 5000, 1000, step=100)
        self._spin_overlap = _spin(0,   4900,  500, step=100)
        wn.addRow(_lbl("Janela (ms):"),      self._spin_window)
        wn.addRow(_lbl("Sobreposição (ms):"), self._spin_overlap)
        v.addWidget(win_box)

        v.addStretch()
        return w

    # ── Aba: Dashboard ────────────────────────────────────────────────────────

    def _tab_dashboard(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setSpacing(12)
        v.setContentsMargins(12, 12, 12, 12)

        # Tipo
        type_box = QtWidgets.QGroupBox("Tipo de dashboard")
        tv = QtWidgets.QVBoxLayout(type_box)
        self._rb_interactive = QtWidgets.QRadioButton(
            "Interativo (PyQtGraph) — painéis ajustáveis em tempo real  ✦ recomendado"
        )
        self._rb_static = QtWidgets.QRadioButton(
            "Estático (matplotlib) — painéis fixos"
        )
        self._rb_interactive.setChecked(True)
        for rb in (self._rb_interactive, self._rb_static):
            rb.setStyleSheet(f"color: {TEXT}; spacing: 6px;")
            tv.addWidget(rb)
        v.addWidget(type_box)

        # Painéis do dashboard estático
        self._grp_panels = QtWidgets.QGroupBox(
            "Painéis iniciais (dashboard estático)"
        )
        pv = QtWidgets.QVBoxLayout(self._grp_panels)
        self._panel_cbs = {}
        for key, label, default in [
            ("emg",     "Sinal EMG",               True),
            ("class",   "Classe predita",           True),
            ("raster",  "Spike raster",             True),
            ("spikes",  "Contagem de spikes",       True),
            ("history", "Histórico de classificação", True),
            ("labels",  "True Labels",              False),
        ]:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(default)
            pv.addWidget(cb)
            self._panel_cbs[key] = cb
        self._grp_panels.setVisible(False)
        v.addWidget(self._grp_panels)

        self._rb_static.toggled.connect(self._grp_panels.setVisible)
        self._rb_interactive.toggled.connect(
            lambda on: self._grp_panels.setVisible(not on)
        )

        v.addStretch()
        return w

    # ── Log ──────────────────────────────────────────────────────────────────

    @QtCore.pyqtSlot(str, str)
    def _append_log(self, text, level):
        colors = {"info": TEXT, "error": RED, "warn": YELLOW}
        color  = colors.get(level, TEXT)
        # Destaca linhas de aviso e erro automaticamente
        if any(w in text.upper() for w in ("ERRO", "ERROR", "FALHA", "FAILED")):
            color = RED
        elif any(w in text.upper() for w in ("AVISO", "WARN", "WARNING")):
            color = YELLOW
        elif any(w in text.upper() for w in ("OK", "CONFIGURADO", "CONECTADO", "INICIADO")):
            color = GREEN

        self._log.append(f'<span style="color:{color};">{text}</span>')
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    # ── Browse helper ─────────────────────────────────────────────────────────

    def _browse_file(self, field: QtWidgets.QLineEdit, filter_str: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecionar arquivo",
            os.path.expanduser("~"),
            f"{filter_str};;Todos os arquivos (*)",
        )
        if path:
            field.setText(path)

    # ── Montar argumentos e iniciar pipeline ──────────────────────────────────

    def _build_args(self):
        """Constrói um namespace compatível com o argparse do real_time_run.py."""
        args = types.SimpleNamespace()

        # Fonte de dados
        args.live        = self._rb_live.isChecked()
        args.mat         = self._mat_path.text().strip() if self._rb_mat.isChecked() else None
        args.signal      = self._mat_signal.text().strip() or "emg"
        args.label       = self._mat_label.text().strip()  or "stimulus"

        # Hardware
        args.fs_hw       = int(self._cb_fs_hw.currentText())
        args.no_realtime = False

        # Modelo
        args.weights     = self._weights_path.text().strip() or None
        args.channels    = self._spin_channels.value()
        args.classes     = self._spin_classes.value()
        args.steps       = self._spin_steps.value()
        args.fs          = self._spin_fs.value()

        # Dashboard
        args.interactive = self._rb_interactive.isChecked()
        args.panels      = (
            [k for k, cb in self._panel_cbs.items() if cb.isChecked()]
            if self._rb_static.isChecked() else None
        )

        return args

    def _validate(self, args):
        """Retorna lista de erros de validação (vazia = ok)."""
        errs = []
        if args.mat and not os.path.isfile(args.mat):
            errs.append(f"Arquivo .mat não encontrado: {args.mat}")
        if args.weights and not os.path.isfile(args.weights):
            errs.append(f"Arquivo de pesos não encontrado: {args.weights}")
        if self._spin_overlap.value() >= self._spin_window.value():
            errs.append("Sobreposição deve ser menor que o tamanho da janela.")
        return errs

    def _start_pipeline(self):
        args = self._build_args()
        errs = self._validate(args)
        if errs:
            QtWidgets.QMessageBox.critical(
                self, "Erro de configuração", "\n".join(errs)
            )
            return

        self._btn_start.setEnabled(False)
        self._btn_start.setText("⏳  Executando…")
        self._log.clear()
        self._append_log("Iniciando pipeline…", "info")

        # Injeta window_ms/overlap_ms nos defaults do pipeline
        from real_time_run import DEFAULT_CONFIG
        DEFAULT_CONFIG["window_ms"]  = self._spin_window.value()
        DEFAULT_CONFIG["overlap_ms"] = self._spin_overlap.value()

        # Injeta configurações de hardware no LiveEMGStream via defaults
        if args.live:
            two_devices = self._spin_devices.currentIndex() == 1
            args._active_devices = [1, 2] if two_devices else [1]
            args._gain           = int(self._cb_gain.currentText())

        def _run():
            try:
                import real_time_run
                # Sobrescreve os hardcodes do arquivo original
                real_time_run.main.__globals__["_LAUNCHER_ARGS"] = args
                _patch_main_args(args)
                real_time_run.main()
            except SystemExit:
                pass
            except Exception:
                sys.stderr.write(traceback.format_exc())
            finally:
                QtCore.QMetaObject.invokeMethod(
                    self, "_on_pipeline_finished", QtCore.Qt.QueuedConnection
                )

        self._pipeline_thread = threading.Thread(target=_run, daemon=True)
        self._pipeline_thread.start()

    @QtCore.pyqtSlot()
    def _on_pipeline_finished(self):
        self._btn_start.setEnabled(True)
        self._btn_start.setText("▶  Iniciar")
        self._append_log("Pipeline encerrado.", "info")

    # ── Restaura stdout/stderr ao fechar ─────────────────────────────────────

    def closeEvent(self, event):
        sys.stdout = self._log_out._orig
        sys.stderr = self._log_err._orig
        super().closeEvent(event)


# ── Patch: injeta args do launcher no main() sem modificar real_time_run.py ──

def _patch_main_args(args):
    """
    Substitui o bloco de hardcodes e parse_args() do real_time_run.py
    pelo namespace construído pelo launcher, sem alterar o arquivo original.
    """
    import real_time_run as rtr

    _orig_main = rtr.main

    def _patched_main():
        import argparse

        # Guarda o parse_args original e o substitui temporariamente
        _orig_parse = argparse.ArgumentParser.parse_args

        def _fake_parse(self_parser, *a, **kw):
            return args

        argparse.ArgumentParser.parse_args = _fake_parse

        # Injeta ganho e devices no LiveEMGStream quando modo ao vivo
        if getattr(args, "live", False):
            import live_emg_stream as lems
            _orig_init = lems.LiveEMGStream.__init__

            def _patched_init(self_s, *a, **kw):
                kw.setdefault("active_devices", getattr(args, "_active_devices", [1]))
                kw.setdefault("gain",           getattr(args, "_gain", 24))
                _orig_init(self_s, *a, **kw)

            lems.LiveEMGStream.__init__ = _patched_init

        try:
            _orig_main()
        finally:
            argparse.ArgumentParser.parse_args = _orig_parse

    rtr.main = _patched_main


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
