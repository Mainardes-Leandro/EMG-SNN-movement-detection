import serial
import time


class HandBridge:
    def __init__(self, port: str, baudrate: int, enabled: bool = True, timeout: float = 0.2):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled
        self.timeout = timeout
        self.ser = None
        
        # Temporização e histerese
        self.cooldown_s = 2
        self.last_send_t = 0.0
        self.last_esc = 0
        self.is_gripping = False
        self.reflex_active = False  # Reflexo só atua depois de um fechamento manual
        
        # Abertura e aperto progressivo
        self.base_pos = 10      # Posição base da mão em repouso
        self.current_pos = 50   # Posição atual usada pelo reflexo
        self.pos_increment = 3  # Incremento aplicado quando há slip
        self.max_pos = 100      # Limite máximo de posição aceito pela mão
        self.fixed_speed = 200  # Velocidade fixa usada nos reflexos

        # Freio do motor
        self.brake_delay_s = 0.35  # Espera antes de enviar o freio B1
        self.brake_sent = True     # Indica se o freio já foi enviado

        # Relaxamento adaptativo
        self.relax_delay_s = 20000.0  # Tempo sem slip antes de começar a afrouxar
        self.relax_step_s = 0.5       # Intervalo entre passos de relaxamento
        self.relax_decrement = 5      # Quanto a posição recua em cada passo
        self.last_slip_t = 0.0        # Instante do último slip detectado
        self.last_relax_t = 0.0       # Instante do último relaxamento aplicado

        self.last_cmd = "Nenhum"

    def connect(self):
        if not self.enabled:
            print("[HandBridge] AVISO: HandBridge desativado (HAND_ENABLE=False).")
            return
        
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.0)
            print(f"[HandBridge] Conectado à mão robótica na porta {self.port} a {self.baudrate} bps.")
        except Exception as e:
            print(f"[HandBridge] AVISO: Falha ao conectar na porta {self.port}. A mão NÃO será controlada.")
            print(f"[HandBridge] Exceção: {e}")
            self.ser = None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[HandBridge] Conexão com a mão robótica encerrada.")

    def send(self, cmd: str):
        self.last_cmd = cmd.strip()
        if not self.enabled or self.ser is None or not self.ser.is_open:
            return
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.ser.write(cmd.encode('utf-8'))
                self.ser.flush()
                time.sleep(0.1)
                print(f"[HandBridge] Enviado: {cmd.strip()}")
                return
            except (PermissionError, OSError) as e:
                print(f"[HandBridge] Erro ao enviar comando (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    try:
                        if self.ser and not self.ser.is_open:
                            self._reconnect()
                    except Exception:
                        pass
                else:
                    print(f"[HandBridge] AVISO: Falha persistente ao enviar. Comando descartado.")
            except Exception as e:
                print(f"[HandBridge] Erro inesperado ao enviar comando: {e}")
                return

    def _reconnect(self):
        """Tenta reconectar a porta serial após um erro."""
        print("[HandBridge] Tentando reconectar...")
        try:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.0)
            print(f"[HandBridge] Reconectado com sucesso na porta {self.port}.")
        except Exception as e:
            print(f"[HandBridge] Falha ao reconectar: {e}")
            self.ser = None

    def large_diameter_grasp(self, pos: int = 50, speed: int = None):
        speed_str = f" S{speed}" if speed is not None else ""
        self.send(f"F0 P{pos}{speed_str}\n")
        self.send(f"F2 P{pos}{speed_str}\n")
        self.send(f"F1 P{pos}{speed_str}\n")
        self.send(f"F3 P{pos}{speed_str}\n")
        self.reflex_active = True
        self.current_pos = pos

    def power_disk_grasp(self, pos: int = 20, thumb_pos: int = 50, speed: int = None):
        """Fecha a pinça com polegar e indicador."""
        speed_str = f" S{speed}" if speed is not None else ""
        self.send(f"F0 P{thumb_pos}{speed_str}\n")
        self.send(f"F2 P{pos}{speed_str}\n")
        self.send(f"F1 P{pos}{speed_str}\n")
        self.send(f"F3 P{pos}{speed_str}\n")
        self.reflex_active = True
        self.current_pos = pos
    
    def tip_pinch_grasp(self, pos: int = 100, thumb_pos: int = 50, speed: int = None):
        """Fecha a pinça com polegar e indicador."""
        speed_str = f" S{speed}" if speed is not None else ""
        self.send(f"F0 P{thumb_pos}{speed_str}\n")
        self.send(f"F1 P{pos}{speed_str}\n")
        self.send(f"F2 P0{speed_str}\n")
        self.send(f"F3 P0{speed_str}\n")
        self.reflex_active = True
        self.current_pos = pos

    def rest(self, speed: int = None):
        """Abre a mão de volta a posição inicial."""
        speed_str = f" S{speed}" if speed is not None else ""
        self.send(f"F0 P0{speed_str}\n")
        self.send(f"F2 P0{speed_str}\n")
        self.send(f"F1 P0{speed_str}\n")
        self.send(f"F3 P0{speed_str}\n")
        self.reflex_active = False
        self.current_pos = self.base_pos
