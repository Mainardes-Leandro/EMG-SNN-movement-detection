"""
snn_model.py
------------
Modelo SNN com neurônios LIF (Leaky Integrate-and-Fire) usando snnTorch.
Recebe janelas de EMG pré-processadas e gera spike trains por classe.

Arquitetura:
  [n_channels * n_features] → FC → LIF → FC → LIF → [n_classes]
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np

def delta_encoding_up_down(x: torch.Tensor, theta, alpha: float = 0.001) -> torch.Tensor:
        """
        x shape: [samples, T, channels]
        theta: escalar ou tensor [channels]
        """
        samples, T, C = x.shape
        
        spikes_up   = torch.zeros(samples, T, 8)
        spikes_down = torch.zeros(samples, T, 8)

        # garante shape [1, 1, C] para broadcast correto
        if isinstance(theta, torch.Tensor) and theta.ndim == 1:
            theta = theta.unsqueeze(0).unsqueeze(0)  # [1, 1, C]

        x_ref = x[:, 0, :8].clone()  # [samples, C]

        for t in range(1, T):
            delta = x[:, t, :8] - x_ref           # [samples, C]
            
            up_mask   = delta >  theta.squeeze(0)  # [samples, C]
            down_mask = delta < -theta.squeeze(0)

            spikes_up[:, t, :][up_mask]     = 1
            spikes_down[:, t, :][down_mask] = 1
            
            reset_mask = up_mask | down_mask
            leak_mask  = ~reset_mask

            x_ref[reset_mask] = x[:, t, :8][reset_mask]
            x_ref[leak_mask]  = (1 - alpha) * x_ref[leak_mask] + alpha * x[:, t, :8][leak_mask]

        return torch.cat([spikes_up, spikes_down], dim=2)  # [samples, T, 2*C]

class EMGSpikeEncoder:
    """
    Converte uma janela de EMG [window_samples, n_channels] em
    features de entrada para o SNN.

    Features por canal (8 no total):
      - RMS, MAV, WL, ZC, SSC, VAR, IEMG, MNF
    """

    def __init__(self, fs: int = 2000):
        self.fs = fs

    def extract(self, window: np.ndarray) -> np.ndarray:
        """
        window: [n_samples, n_channels]
        retorna: [n_channels * n_features]
        """
        n_samples, n_channels = window.shape
        features = []

        for ch in range(n_channels):
            x = window[:, ch]

            rms   = np.sqrt(np.mean(x ** 2))
            mav   = np.mean(np.abs(x))
            wl    = np.sum(np.abs(np.diff(x)))
            zc    = np.sum(np.diff(np.sign(x)) != 0) / n_samples
            ssc   = np.sum(np.diff(np.sign(np.diff(x))) != 0) / n_samples
            var   = np.var(x)
            iemg  = np.sum(np.abs(x))

            # Mean frequency (via FFT)
            fft_mag = np.abs(np.fft.rfft(x))
            freqs   = np.fft.rfftfreq(n_samples, 1 / self.fs)
            mnf     = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-8)

            features.extend([rms, mav, wl, zc, ssc, var, iemg, mnf])

        return np.array(features, dtype=np.float32)


class EMGSNN(nn.Module):
    """
    SNN com duas camadas de neurônios LIF para classificação de gestos EMG.

    Parâmetros
    ----------
    n_inputs   : dimensão do vetor de features de entrada
    n_hidden   : neurônios na camada oculta
    n_classes  : número de classes de movimento
    n_steps    : número de passos de tempo para integração de spikes
    beta       : fator de decaimento do potencial de membrana (0-1)
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 beta=0.99, learn_beta=True, learn_threshold=False, slope=25.0):
        super().__init__()

        spike_grad = snn.surrogate.fast_sigmoid(slope=slope)

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=learn_beta,
                              learn_threshold=learn_threshold,
                              spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta,
                              learn_threshold=learn_threshold,
                              spike_grad=spike_grad)

        # readout linear (não spiking) -> produz logits para CrossEntropy
        self.readout = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
    
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
    
        mem_rec = []
    
        for t in range(x.size(0)):
    
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
    
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
    
            mem_rec.append(mem2)
    
        mem_out = torch.stack(mem_rec)
        mem_out = mem_out.mean(0)
    
        logits = self.readout(mem_out)
        
        return logits

    def predict(self, x: torch.Tensor):
        """
        x: [T, 2*C] — spike trains ao longo do tempo
        """
        with torch.no_grad():
            logits = self(x)  # [n_classes]

        pred_class  = logits.argmax(dim=0)              # escalar
        confidence  = torch.softmax(logits, dim=0)      # [n_classes] — usado como spike_count
        spike_rec   = logits.unsqueeze(0).unsqueeze(0)  # [1, 1, n_classes] — dummy pro dashboard

        return pred_class.unsqueeze(0), confidence.unsqueeze(0), spike_rec


def build_model(n_channels: int = 8, n_classes: int = 4, n_steps: int = 25) -> tuple:
    n_inputs = n_channels * 2  # up + down por canal (após agregação temporal)

    model = EMGSNN(
        input_size=n_inputs,
        hidden_size1=128,
        hidden_size2=64,
        output_size=n_classes
    )
    model.eval()
    return model

def load_pretrained(model: EMGSNN, path: str) -> EMGSNN:
    """Carrega pesos salvos de um checkpoint."""
    state = torch.load(path, map_location="cpu")
    train_mean = state["train_mean"]
    train_std = state["train_std"]
    theta = state["theta"]

    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[SNN] Pesos carregados de '{path}'")
    return model, train_mean, train_std, theta