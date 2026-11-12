"""
main.py
-------
Pipeline principal: EMG stream → SNN → Dashboard ao vivo.

Uso rápido (dados sintéticos):
    python main.py

Com arquivo .mat real:
    python main.py --mat /caminho/dados.mat --signal emg --label label

Com modelo pré-treinado:
    python main.py --mat /caminho/dados.mat --weights /caminho/model.pt
"""

import argparse
import numpy as np
import torch
import time
import threading                          # [LIVE]

from dataloader import SyntheticEMGGenerator, EMGMatLoader
from snn_model   import build_model, load_pretrained, delta_encoding_up_down
from dashboard   import StreamingDashboard
from dashboard_interactive import InteractiveDashboard  # [INTERACTIVE]
from hand_bridge import HandBridge
from live_emg_stream import LiveEMGStream  # [LIVE]

# ──────────────────────────────────────────────────────────────
# Configuração padrão do pipeline
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    n_channels   = 8,
    n_classes    = 4,
    n_steps      = 25,
    fs           = 200,
    window_ms    = 1000,
    overlap_ms   = 500,
    realtime     = True,   # False = sem delay (o mais rápido possível)
)


def run_pipeline(stream, model, theta, dashboard, cfg, CLASS_MAP, hand, train_mean, train_std):
    last_cmd = None
    for window, label, timestamp in stream:
        if not dashboard.running:
            break
        train_mean_arr = np.asarray(train_mean, dtype=np.float64)
        train_std_arr  = np.asarray(train_std,  dtype=np.float64)
        window_norm = (window[:, :8] - train_mean_arr) / (train_std_arr + 1e-8)

        # 1. Prepara tensor [1, T, C]
        window_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)

        # 2. Delta encoding → [1, T, 2*C]
        spikes = delta_encoding_up_down(window_tensor, theta=theta, alpha=0.017)

        # 3. Agrega no tempo → [1, 2*C]
        spike_rates = spikes.squeeze(0)

        # 4. Inferência SNN
        pred_class, spike_count, spike_rec = model.predict(spike_rates)

        CMD_DELAY = 1.0  # segundos

        cmd = CLASS_MAP[int(pred_class[0])]

        confidence = spike_count[0].max().item()

        if confidence > 0.5:
            if cmd != last_cmd:
                if cmd == "rest":
                    hand.rest()

                elif cmd == "power_disk_grasp":
                    hand.power_disk_grasp(pos=20)

                elif cmd == "tip_pinch_grasp":
                    hand.tip_pinch_grasp(pos=100, thumb_pos=60)

                elif cmd == "large_diameter_grasp":
                    hand.large_diameter_grasp(pos=60)

                last_cmd = cmd

        # 5. Envia ao dashboard
        dashboard.push(
            emg_window  = window,
            pred_class  = int(pred_class[0]),
            spike_count = spike_count,
            spike_rec   = spikes[0],   # [T, 2*C] para o raster
            timestamp   = timestamp,
            true_label  = label
        )

def main():
    parser = argparse.ArgumentParser(description="EMG → SNN Streaming Pipeline")
    parser.add_argument("--mat",     type=str,   default=None,  help="Caminho para o arquivo .mat")
    parser.add_argument("--signal",  type=str,   default="emg", help="Chave do sinal no .mat")
    parser.add_argument("--label",   type=str,   default="label", help="Chave dos labels no .mat")
    parser.add_argument("--weights", type=str,   default=None,  help="Caminho para pesos do modelo (.pt)")
    parser.add_argument("--no-realtime", action="store_true",   help="Remove delay de tempo real")
    parser.add_argument("--channels", type=int,  default=DEFAULT_CONFIG["n_channels"])
    parser.add_argument("--classes",  type=int,  default=DEFAULT_CONFIG["n_classes"])
    parser.add_argument("--steps",    type=int,  default=DEFAULT_CONFIG["n_steps"])
    parser.add_argument("--fs",       type=int,  default=DEFAULT_CONFIG["fs"])
    # [LIVE] ──────────────────────────────────────────────────────────────────
    parser.add_argument("--live",     action="store_true", help="Usa sinal ao vivo do hardware (ESP32/ADS1299)")
    parser.add_argument("--fs-hw",    type=int,  default=1000,  help="Taxa do hardware em SPS (padrão: 1000)")
    parser.add_argument(
        "--panels", nargs="+",
        default=None,
        choices=["emg", "class", "raster", "spikes", "history", "labels"],
        metavar="PAINEL",
        help=(
            "Painéis a exibir (dashboard estático). Opções: emg class raster spikes history labels. "
            "Padrão: todos. Exemplo: --panels emg class history"
        ),
    )
    parser.add_argument(              # [INTERACTIVE]
        "--interactive", action="store_true",
        help="Usa o dashboard interativo (PyQtGraph) com checkboxes para alternar painéis em tempo real.",
    )
    args = parser.parse_args()
    
    args.mat = "S1_E3_A1.mat"
    args.weights = "modelo_snn_v3.pth"
    args.channels = 8
    args.classes = 2
    args.fs = 200
    args.no_realtime = False
    args.steps = 10
    args.label = "stimulus"
    args.live = True
    args.interactive = True

    cfg = DEFAULT_CONFIG.copy()
    cfg["n_channels"] = args.channels
    cfg["n_classes"]  = args.classes
    cfg["n_steps"]    = args.steps
    cfg["fs"]         = args.fs
    cfg["realtime"]   = not args.no_realtime

    # ── Fonte de dados ─────────────────────────────────────────
    # [LIVE] ──────────────────────────────────────────────────────────────────
    if args.live:
        print(f"[main] Modo ao vivo — hardware ESP32/ADS1299 @ {args.fs_hw} SPS")
        live_stream = LiveEMGStream(
            n_channels    = cfg["n_channels"],
            fs_hw         = args.fs_hw,
            fs_target     = cfg["fs"],
            window_ms     = cfg["window_ms"],
            overlap_ms    = cfg["overlap_ms"],
            gain          = 1,     
            convert_to_mv = True,   # NinaPro está em mV; False se treinou em contagens brutas
        )
        stream = live_stream.stream()
    elif args.mat:
        print(f"[main] Carregando arquivo: {args.mat}")
        loader = EMGMatLoader(
            mat_path   = args.mat,
            signal_key = args.signal,
            label_key  = args.label,
            fs         = cfg["fs"],
            window_ms  = cfg["window_ms"],
            overlap_ms = cfg["overlap_ms"],
        )
        cfg["n_channels"] = 8 #loader.n_channels
        stream = loader.stream(realtime=cfg["realtime"])
    else:
        print("[main] Nenhum .mat fornecido — usando dados SINTÉTICOS para teste.")
        gen = SyntheticEMGGenerator(
            n_channels = cfg["n_channels"],
            fs         = cfg["fs"],
            window_ms  = cfg["window_ms"],
            overlap_ms = cfg["overlap_ms"],
            duration_s = 30.0,
        )
        stream = gen.stream(realtime=cfg["realtime"])

    # ── Modelo SNN ─────────────────────────────────────────────
    #theta = torch.tensor([0.12] * cfg["n_channels"], dtype=torch.float32)
    
    model = build_model(
        n_channels = cfg["n_channels"],
        n_classes  = cfg["n_classes"],
        n_steps    = cfg["n_steps"],
    )

    if args.weights:
        model, train_mean, train_std, theta = load_pretrained(model, args.weights)
    else:
        print("[main] Usando pesos ALEATÓRIOS (modelo não treinado).")
        print("       Para usar um modelo treinado: --weights caminho/modelo.pt")

    # ── Dashboard ──────────────────────────────────────────────
    if args.interactive:                                       # [INTERACTIVE]
        dashboard = InteractiveDashboard(
            n_channels  = cfg["n_channels"],
            n_classes   = cfg["n_classes"],
            n_steps     = cfg["n_steps"],
            history_len = 200,
            update_ms   = 100,
        )
    else:
        dashboard = StreamingDashboard(
            n_channels  = cfg["n_channels"],
            n_classes   = cfg["n_classes"],
            n_steps     = cfg["n_steps"],
            history_len = 80,
            update_ms   = 200,
            panels      = set(args.panels) if args.panels else None,
        )

    hand = HandBridge(
    port="COM5", 
    baudrate=115200,
    enabled=True
    )
    hand.connect()

    CLASS_MAP = {
    0: "rest",
    1: "large_diameter_grasp",
    2: "tip_pinch_grasp",
    3: "power_disk_grasp"
    }
    
    # ── Inicia tudoc:\Users\maina\Downloads\mechanoreceptor\EMG_data\s2\S2_E2_A1.mat ────────────────────────────────────────────
    pipeline_fn = lambda: run_pipeline(stream, model, theta, dashboard, cfg, CLASS_MAP, hand, train_mean, train_std)

    # [LIVE] Thread que aguarda Enter e encerra a aquisição de forma limpa.
    # Iniciada aqui para garantir que `dashboard` já existe quando for referenciado.
    if args.live:
        def _wait_for_enter():
            input("\nPressione Enter para encerrar a aquisição...\n")
            print("[main] Encerrando...")
            live_stream.stop()
            dashboard.running = False
        threading.Thread(target=_wait_for_enter, daemon=True).start()

    dashboard.start(pipeline_fn)
    hand.close()


if __name__ == "__main__":
    main()