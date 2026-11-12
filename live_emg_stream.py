"""
live_emg_stream.py
------------------
Fonte de dados ao vivo para o pipeline EMG → SNN.

Implementa a mesma interface de generator que EMGMatLoader e
SyntheticEMGGenerator: .stream() yields (window, label, timestamp),
onde label é sempre None (sem ground truth no dado ao vivo).

Uso direto (teste sem o pipeline completo):
    stream = LiveEMGStream(fs_hw=1000, fs_target=200, window_ms=1000, overlap_ms=500)
    for window, label, ts in stream.stream():
        print(window.shape, ts)
"""

import socket
import threading
import time
import numpy as np
from collections import deque

try:
    from scipy.signal import resample
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Constantes do protocolo ESP32/ADS1299 ────────────────────────────────────
_HOST       = "192.168.4.1"
_PORT       = 3333
_FRAME_SIZE = 27 * 8        # 216 bytes: 8 AFEs em daisy-chain
_ACTIVE_AFE = slice(0, 27)  # ADS1299 #1 confirmado experimentalmente

# ── Constantes de configuração de hardware ────────────────────────────────────
# Registradores
_CH_REGISTERS      = [0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]
_CONFIG1           = 0x01
_MISC1             = 0x15
_POWER_DOWN_BYTE   = 0x81   # PD=1, GAIN=000, MUX=001 (curto interno)
_DEVICES_TO_DISABLE = range(2, 9)

# Códigos de taxa de amostragem (bits DR[2:0] do CONFIG1)
_DR_CODES = {16000: 0b000, 8000: 0b001, 4000: 0b010, 2000: 0b011,
             1000:  0b100, 500:  0b101,  250:  0b110}

# Códigos de ganho (bits GAIN[2:0] do CHnSET)
_GAIN_CODES = {1: 0b000, 2: 0b001, 4: 0b010, 6: 0b011,
               8: 0b100, 12: 0b101, 24: 0b110}

# Fator de conversão int24 → mV:
#   valor_mV = contagem × (2 × Vref) / (GAIN × 2^24) × 1000
# Para Vref = 2.5V (referência interna do ADS1299) e GAIN = 24:
#   = contagem × 5 / (24 × 16_777_216) × 1000
#   ≈ contagem × 1.2457 × 10⁻⁵  mV
# O valor correto de SCALE_TO_MV depende do ganho configurado.
# Ele é recalculado em _build_scale_factor() com base em self.gain.
_VREF = 2.5


def _to_int24(b3: bytes) -> float:
    val = int.from_bytes(b3, byteorder="big", signed=False)
    return float(val - 0x1000000 if val & 0x800000 else val)


def _parse_frame(raw216: bytes) -> np.ndarray:
    """Extrai os 8 canais do AFE ativo. Retorna array (8,) em float32."""
    afe = raw216[_ACTIVE_AFE]
    return np.array([_to_int24(afe[3 + 3*i: 6 + 3*i]) for i in range(8)],
                    dtype=np.float32)


# ── Classe principal ──────────────────────────────────────────────────────────

class LiveEMGStream:
    """
    Parâmetros
    ----------
    host, port      : endereço do ESP32
    n_channels      : canais a usar (até 8)
    fs_hw           : taxa de amostragem a CONFIGURAR no hardware (SPS).
                      Deve ser um dos valores: 250, 500, 1000, 2000, 4000, 8000, 16000.
    fs_target       : taxa esperada pelo modelo (reamostrado se != fs_hw)
    window_ms       : duração da janela em ms
    overlap_ms      : sobreposição entre janelas em ms
    gain            : ganho dos canais ativos (1,2,4,6,8,12,24).
                      Use o mesmo valor com que o modelo foi treinado no NinaPro.
    convert_to_mv   : se True, converte contagens int24 → mV antes de retornar.
                      Ative quando train_mean/train_std do seu modelo foram
                      computados em mV (NinaPro). Desative se foram em contagens brutas.
    """

    def __init__(
        self,
        host: str       = _HOST,
        port: int       = _PORT,
        n_channels: int = 8,
        fs_hw: int      = 1000,
        fs_target: int  = 200,
        window_ms: int  = 1000,
        overlap_ms: int = 500,
        gain: int       = 24,
        convert_to_mv: bool = True,
    ):
        if overlap_ms >= window_ms:
            raise ValueError("overlap_ms deve ser menor que window_ms")
        if fs_hw not in _DR_CODES:
            raise ValueError(f"fs_hw inválido. Opções: {sorted(_DR_CODES)} SPS")
        if gain not in _GAIN_CODES:
            raise ValueError(f"ganho inválido. Opções: {sorted(_GAIN_CODES)}")

        self.host          = host
        self.port          = port
        self.n_channels    = n_channels
        self.fs_hw         = fs_hw
        self.fs_target     = fs_target
        self.window_ms     = window_ms
        self.overlap_ms    = overlap_ms
        self.gain          = gain
        self.convert_to_mv = convert_to_mv

        # Fator de conversão: contagens → mV
        # valor_mV = contagem × (2 × Vref × 1000) / (GAIN × 2^24)
        self._scale = (2 * _VREF * 1000) / (gain * 2**24) if convert_to_mv else 1.0

        self._stop_event  = threading.Event()
        self._ctrl_sock: socket.socket | None = None
        self._data_sock: socket.socket | None = None
        self._sample_buf: deque               = deque()
        self._buf_lock    = threading.Lock()

    # ── API pública ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Para a aquisição. Pode ser chamado de qualquer thread."""
        self._stop_event.set()
        if self._ctrl_sock:
            try:
                self._ctrl_sock.sendall(b"stop\n")
            except OSError:
                pass
        if self._data_sock:
            try:
                self._data_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def stream(self):
        """
        Generator que yields (window, None, timestamp).

        window    : np.ndarray shape (window_samples_target, n_channels), float32
                    Valores em mV se convert_to_mv=True (default), em contagens brutas caso contrário.
        label     : sempre None
        timestamp : float, segundos desde o início da aquisição
        """
        win_hw  = int(self.window_ms  / 1000 * self.fs_hw)
        step_hw = int((self.window_ms - self.overlap_ms) / 1000 * self.fs_hw)
        win_tgt = int(self.window_ms  / 1000 * self.fs_target)

        print(f"[LiveEMGStream] Conectando a {self.host}:{self.port}...")
        self._ctrl_sock = socket.create_connection((self.host, self.port), timeout=10)
        print("[LiveEMGStream] Conectado (socket de controle).")

        self._configure_hardware()

        resp = self._send_ctrl("sample -t 86400")
        if resp != "0":
            self._ctrl_sock.close()
            raise RuntimeError(f"[LiveEMGStream] Falha ao iniciar aquisição: {resp!r}")

        # Segunda conexão — dispara o START físico dos ADS1299
        self._data_sock = socket.create_connection((self.host, self.port))
        self._data_sock.settimeout(None)
        print("[LiveEMGStream] Conectado (socket de dados). Streaming iniciado.")
        print(f"[LiveEMGStream] Aguardando primeira janela "
              f"({win_hw} amostras @ {self.fs_hw} SPS = {self.window_ms} ms)...")

        reader = threading.Thread(target=self._reader_thread, daemon=True)
        reader.start()

        t0 = time.time()
        first_window = True

        try:
            while not self._stop_event.is_set():
                # Aguarda amostras suficientes para uma janela
                while True:
                    with self._buf_lock:
                        available = len(self._sample_buf)
                    if available >= win_hw:
                        break
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.005)

                with self._buf_lock:
                    raw = np.array(
                        [self._sample_buf[i] for i in range(win_hw)],
                        dtype=np.float32,
                    )  # (win_hw, 8)
                    for _ in range(step_hw):
                        self._sample_buf.popleft()

                # Converte escala
                raw = raw * self._scale  # noqa: E226 (broadcasting, sem cópia extra)

                # Reamostra de fs_hw → fs_target
                window = self._resample(raw, win_tgt)
                window = window[:, :self.n_channels]

                timestamp = time.time() - t0
                if first_window:
                    print(
                        f"[LiveEMGStream] Primeira janela gerada — shape {window.shape}, "
                        f"min={window.min():.4f}, max={window.max():.4f} "
                        f"({'mV' if self.convert_to_mv else 'counts'})."
                    )
                    first_window = False

                yield window, None, timestamp

        finally:
            print("[LiveEMGStream] Encerrando...")
            self.stop()
            reader.join(timeout=3)
            for s in (self._ctrl_sock, self._data_sock):
                if s:
                    try:
                        s.close()
                    except OSError:
                        pass

    # ── Configuração de hardware ──────────────────────────────────────────────

    def _configure_hardware(self) -> None:
        """
        Configura o ADS1299 #1 para aquisição e desliga os outros 7.
        Chamado automaticamente em stream() antes de iniciar a amostragem.
        """
        import time as _t

        print(f"[LiveEMGStream] Desligando ADS1299 #2–8...")
        for device in _DEVICES_TO_DISABLE:
            for reg in _CH_REGISTERS:
                self._send_ctrl(
                    f"adswreg -r {reg:#04x} -d {device} {_POWER_DOWN_BYTE:#04x}"
                )
                _t.sleep(0.02)

        print(f"[LiveEMGStream] Configurando ADS1299 #1 "
              f"(ganho={self.gain}x, SRB1=ON, {self.fs_hw} SPS)...")

        # Ganho + entrada normal em todos os 8 canais
        ch_byte = (_GAIN_CODES[self.gain] << 4) | 0b000  # MUX=000 (normal)
        for reg in _CH_REGISTERS:
            self._send_ctrl(f"adswreg -r {reg:#04x} -d 1 {ch_byte:#04x}")

        # SRB1: conecta o terminal negativo de todos os canais à referência comum
        self._send_ctrl(f"adswreg -r {_MISC1:#04x} -d 1 0x20")

        # Taxa de amostragem: lê CONFIG1 atual e troca só os bits DR[2:0]
        # (preserva DAISY_EN e CLK_EN que o firmware já configurou)
        cfg1_raw = self._send_ctrl(f"adsrreg -r {_CONFIG1:#04x} -d 1")
        try:
            cfg1 = int(cfg1_raw.strip().lstrip("x").lstrip("0x") or "0", 16)
        except ValueError:
            cfg1 = 0xB6  # fallback: valor padrão observado na dissertação
        new_cfg1 = (cfg1 & 0b11111000) | _DR_CODES[self.fs_hw]
        self._send_ctrl(f"adswreg -r {_CONFIG1:#04x} -d 1 {new_cfg1:#04x}")

        print("[LiveEMGStream] Hardware configurado.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _reader_thread(self) -> None:
        buf = b""
        while not self._stop_event.is_set():
            try:
                chunk = self._data_sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while len(buf) >= _FRAME_SIZE:
                frame, buf = buf[:_FRAME_SIZE], buf[_FRAME_SIZE:]
                sample = _parse_frame(frame)
                with self._buf_lock:
                    self._sample_buf.append(sample)

    def _resample(self, raw: np.ndarray, target_samples: int) -> np.ndarray:
        if raw.shape[0] == target_samples:
            return raw

        factor = raw.shape[0] // target_samples  # 5 com 1000→200

        # Corta o excesso para que raw.shape[0] seja múltiplo exato de factor
        n = (raw.shape[0] // factor) * factor
        return raw[:n].reshape(-1, factor, raw.shape[1]).mean(axis=1).astype(np.float32)

    def _send_ctrl(self, cmd: str) -> str:
        self._ctrl_sock.sendall((cmd + "\n").encode())
        response = b""
        while not response.endswith(b"\n"):
            chunk = self._ctrl_sock.recv(1024)
            if not chunk:
                break
            response += chunk
        return response.decode().strip()
