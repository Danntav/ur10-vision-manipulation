import numpy as np
import math
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import scipy.linalg

# --- 1. INICIALIZAÇÃO E CONEXÃO ---
try:
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    print("Conectado ao CoppeliaSim")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível conectar ao CoppeliaSim. Erro: {e}")
    exit()

# --- 2. HANDLES ---
try:
    joint_handles = [sim.getObject(f'/UR10/joint{i+1}') for i in range(6)]
    print("Handles das juntas obtidos.")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível encontrar as juntas. Erro: {e}")
    exit()

# --- 3. FUNÇÕES DE MÉTRICAS (UNIFICADAS) ---
def calculate_metrics(time_data, angle_history_rad, target_angles_rad, settling_threshold=0.02):
    """
    Calcula RMSE, Settling Time e Overshoot para cada junta.
    Recebe todos os dados em RADIANOS para consistência.
    """
    angle_history_rad = np.array(angle_history_rad)
    target_angles_rad = np.array(target_angles_rad)
    
    n_joints = angle_history_rad.shape[1]
    
    rmse_values = np.zeros(n_joints)
    settling_times = np.zeros(n_joints)
    overshoot = np.zeros(n_joints)
    
    for i in range(n_joints):
        # RMSE
        errors = angle_history_rad[:, i] - target_angles_rad[i]
        rmse_values[i] = np.sqrt(np.mean(errors**2))
        
        # Settling Time (2% do valor final)
        tolerance = abs(target_angles_rad[i]) * settling_threshold
        if tolerance < 0.01: # Mínimo de 0.01 rad para alvos próximos de zero
            tolerance = 0.01
        
        settled = False
        for j in range(len(time_data)):
            if abs(errors[j]) <= tolerance:
                if all(abs(errors[k]) <= tolerance for k in range(j, len(time_data))):
                    settling_times[i] = time_data[j]
                    settled = True
                    break
        
        if not settled:
            settling_times[i] = time_data[-1] # Não estabilizou
        
        # Overshoot
        if abs(target_angles_rad[i]) > 0.01:
            initial_angle = angle_history_rad[0, i]
            max_angle = np.max(angle_history_rad[:, i])
            min_angle = np.min(angle_history_rad[:, i])
            
            if target_angles_rad[i] > initial_angle:
                overshoot[i] = max(0, (max_angle - target_angles_rad[i]) / abs(target_angles_rad[i])) * 100
            else:
                overshoot[i] = max(0, (target_angles_rad[i] - min_angle) / abs(target_angles_rad[i])) * 100
    
    return rmse_values, settling_times, overshoot

def print_metrics_summary(rmse_values, settling_times, overshoot, step_label):
    """ Imprime métricas de forma organizada. """
    print(f"\n{'='*70}")
    print(f"MÉTRICAS DE DESEMPENHO - {step_label}")
    print(f"{'='*70}")
    print(f"{'Junta':<10} {'RMSE (rad)':<15} {'RMSE (°)':<15} {'Ts (s)':<12} {'Overshoot (%)':<15}")
    print(f"{'-'*70}")
    
    for i in range(6):
        rmse_deg = np.degrees(rmse_values[i])
        print(f"Junta {i+1:<3} {rmse_values[i]:>10.4f}      {rmse_deg:>10.4f}        {settling_times[i]:>8.2f}    {overshoot[i]:>10.2f}")
    
    print(f"\nMÉDIAS:")
    print(f"RMSE Médio: {np.mean(rmse_values):.4f} rad ({np.degrees(np.mean(rmse_values)):.4f}°)")
    print(f"Ts Médio: {np.mean(settling_times):.2f} s")
    print(f"Overshoot Médio: {np.mean(overshoot):.2f}%")
    print(f"{'='*70}\n")

def plot_erro_cartesiano_degrau(pid_data, lqr_data):
    """
    Plota o erro Euclidiano total acumulado durante os testes de degrau.
    Calcula a norma do vetor de erro das 6 juntas em cada instante.
    """
    all_time_pid, all_angles_pid, all_targets_deg = pid_data
    all_time_lqr, all_angles_lqr, _ = lqr_data
    
    # --- Calcula erro Euclidiano acumulado para PID ---
    erro_euclidiano_pid = []
    tempo_acumulado_pid = []
    time_offset_pid = 0
    
    for step_idx, (time_data, angle_history) in enumerate(zip(all_time_pid, all_angles_pid)):
        angle_history_rad = np.array(angle_history)
        target_rad = np.radians(np.array(all_targets_deg[step_idx]))
        
        # Calcula erro para cada amostra
        for t_idx in range(len(time_data)):
            erro_vetor = target_rad - angle_history_rad[t_idx]
            # Norma euclidiana convertida para graus
            erro_norm = np.linalg.norm(np.degrees(erro_vetor))
            erro_euclidiano_pid.append(erro_norm)
            tempo_acumulado_pid.append(time_data[t_idx] + time_offset_pid)
        
        time_offset_pid += time_data[-1]
    
    # --- Calcula erro Euclidiano acumulado para LQR ---
    erro_euclidiano_lqr = []
    tempo_acumulado_lqr = []
    time_offset_lqr = 0
    
    for step_idx, (time_data, angle_history) in enumerate(zip(all_time_lqr, all_angles_lqr)):
        angle_history_rad = np.array(angle_history)
        target_rad = np.radians(np.array(all_targets_deg[step_idx]))
        
        # Calcula erro para cada amostra
        for t_idx in range(len(time_data)):
            erro_vetor = target_rad - angle_history_rad[t_idx]
            # Norma euclidiana convertida para graus
            erro_norm = np.linalg.norm(np.degrees(erro_vetor))
            erro_euclidiano_lqr.append(erro_norm)
            tempo_acumulado_lqr.append(time_data[t_idx] + time_offset_lqr)
        
        time_offset_lqr += time_data[-1]
    
    # --- GRÁFICO ÚNICO E FOCADO ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(tempo_acumulado_pid, erro_euclidiano_pid, 'steelblue', 
            linewidth=2.2, label='PID', alpha=0.9)
    ax.plot(tempo_acumulado_lqr, erro_euclidiano_lqr, 'darkorange', 
            linewidth=2.2, label='LQR', alpha=0.9)
    
    # Marca as transições entre degraus
    time_offset = 0
    for step_idx in range(len(all_targets_deg)):
        end_time_segment = time_offset + all_time_pid[step_idx][-1]
        if step_idx < len(all_targets_deg) - 1:
            ax.axvline(x=end_time_segment, color='gray', linestyle=':', 
                      alpha=0.5, linewidth=1.5)
        time_offset = end_time_segment
    
    ax.set_title('Erro Euclidiano Total na Resposta ao Degrau', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erro Euclidiano Total (°)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Ajuste de zoom
    max_error = max(max(erro_euclidiano_pid), max(erro_euclidiano_lqr))
    ax.set_ylim(0, max_error * 1.15)
    
    plt.tight_layout()
    plt.show()
    
    # --- TABELA DE MÉTRICAS ---
    erro_euclidiano_pid = np.array(erro_euclidiano_pid)
    erro_euclidiano_lqr = np.array(erro_euclidiano_lqr)
    
    print("\n" + "="*70)
    print("ESTATÍSTICAS DO ERRO EUCLIDIANO TOTAL - RESPOSTA AO DEGRAU")
    print("="*70)
    print(f"{'Controlador':<15} {'Erro Médio (°)':<20} {'Erro Máximo (°)':<20} {'Erro RMS (°)':<20}")
    print("-"*70)
    print(f"{'PID':<15} {np.mean(erro_euclidiano_pid):<20.4f} "
          f"{np.max(erro_euclidiano_pid):<20.4f} "
          f"{np.sqrt(np.mean(erro_euclidiano_pid**2)):<20.4f}")
    print(f"{'LQR':<15} {np.mean(erro_euclidiano_lqr):<20.4f} "
          f"{np.max(erro_euclidiano_lqr):<20.4f} "
          f"{np.sqrt(np.mean(erro_euclidiano_lqr**2)):<20.4f}")
    print("="*70)
    

def calcular_indices_desempenho(time_data, angle_history_rad, target_angles_rad):
    """
    Calcula os índices de desempenho: IAE, ISE, ITAE, ITSE
    Retorna os valores médios para todas as juntas.
    """
    angle_history_rad = np.array(angle_history_rad)
    target_angles_rad = np.array(target_angles_rad)
    time_data = np.array(time_data)
    
    n_joints = angle_history_rad.shape[1]
    
    # Inicializa arrays para armazenar índices de cada junta
    IAE_joints = np.zeros(n_joints)
    ISE_joints = np.zeros(n_joints)
    ITAE_joints = np.zeros(n_joints)
    ITSE_joints = np.zeros(n_joints)
    
    for i in range(n_joints):
        # Calcula o erro para cada junta
        error = target_angles_rad[i] - angle_history_rad[:, i]
        error_abs = np.abs(error)
        error_squared = error ** 2
        
        # Calcula os índices usando integração numérica (regra do trapézio)
        IAE_joints[i] = np.trapz(error_abs, time_data)
        ISE_joints[i] = np.trapz(error_squared, time_data)
        ITAE_joints[i] = np.trapz(time_data * error_abs, time_data)
        ITSE_joints[i] = np.trapz(time_data * error_squared, time_data)
    
    # Retorna as médias
    return {
        'IAE': np.mean(IAE_joints),
        'ISE': np.mean(ISE_joints),
        'ITAE': np.mean(ITAE_joints),
        'ITSE': np.mean(ITSE_joints),
        'IAE_joints': IAE_joints,
        'ISE_joints': ISE_joints,
        'ITAE_joints': ITAE_joints,
        'ITSE_joints': ITSE_joints
    }

def print_indices_summary(indices, step_label):
    """
    Imprime os índices de desempenho de forma organizada.
    """
    print(f"\n{'='*70}")
    print(f"ÍNDICES DE DESEMPENHO - {step_label}")
    print(f"{'='*70}")
    print(f"{'Junta':<10} {'IAE':<15} {'ISE':<15} {'ITAE':<15} {'ITSE':<15}")
    print(f"{'-'*70}")
    
    for i in range(6):
        print(f"Junta {i+1:<3} {indices['IAE_joints'][i]:>10.4f}    "
              f"{indices['ISE_joints'][i]:>10.4f}    "
              f"{indices['ITAE_joints'][i]:>10.4f}    "
              f"{indices['ITSE_joints'][i]:>10.4f}")
    
    print(f"\nMÉDIAS:")
    print(f"IAE Médio:  {indices['IAE']:.4f}")
    print(f"ISE Médio:  {indices['ISE']:.4f}")
    print(f"ITAE Médio: {indices['ITAE']:.4f}")
    print(f"ITSE Médio: {indices['ITSE']:.4f}")
    print(f"{'='*70}\n")


# --- 4. CONTROLADOR PID (VERSÃO FINAL COM SATURAÇÃO DE VELOCIDADE) ---
def move_to_config_pid(target_angles_deg, duration_sec=5.0):
    """
    Move o robô usando PID com sintonia fina e saturação de velocidade para garantir estabilidade.
    """
    print(f"\nIniciando movimento PID para (graus): {target_angles_deg}")
    target_angles_rad = np.array([math.radians(a) for a in target_angles_deg])

    # --- SINTONIA FINA: 3 NÍVEIS DE GANHOS (PONTO DE PARTIDA RECOMENDADO) ---
    gains_altos = {'Kp': 15.0, 'Ki': 1.5, 'Kd': 0.5}
    gains_medios = {'Kp': 12.0, 'Ki': 1.5, 'Kd': 0.5}
    gains_punho = {'Kp': 10.0, 'Ki': 1.2, 'Kd': 0.2}

    lista_de_ganhos = [gains_altos, gains_altos, gains_altos, gains_medios, gains_medios, gains_punho]

    Kp = np.array([g['Kp'] for g in lista_de_ganhos])
    Ki = np.array([g['Ki'] for g in lista_de_ganhos])
    Kd = np.array([g['Kd'] for g in lista_de_ganhos])

    integral_error = np.zeros(6)
    previous_error = np.zeros(6)
    
    time_log, angle_log_rad = [], []

    start_time = time.time()
    last_time = start_time
    
    while time.time() - start_time < duration_sec:
        current_time = time.time()
        dt = current_time - last_time
        
        if dt <= 0:
            time.sleep(0.001)
            continue

        current_angles_rad = np.array([sim.getJointPosition(h) for h in joint_handles])
        error = target_angles_rad - current_angles_rad
        
        integral_error += error * dt
        derivative_error = (error - previous_error) / dt
        
        target_velocities = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
        
        # --- LINHA CRÍTICA: SATURAÇÃO DA VELOCIDADE ---
        # Limitamos a velocidade máxima para um valor realista (ex: 90 graus/s ou pi/2 rad/s)
        # Isso impede que o controlador envie comandos impossíveis para a simulação.
        max_vel_rad_s = 2 * np.pi 
        target_velocities = np.clip(target_velocities, -max_vel_rad_s, max_vel_rad_s)
        # --- FIM DA ALTERAÇÃO ---

        for i in range(6):
            sim.setJointTargetVelocity(joint_handles[i], target_velocities[i])
        
        previous_error = error
        last_time = current_time
        
        time_log.append(time.time() - start_time)
        angle_log_rad.append(current_angles_rad.copy())
        
        time.sleep(0.005)

    [sim.setJointTargetVelocity(h, 0) for h in joint_handles]
    print("Movimento PID finalizado.")
    return time_log, angle_log_rad, target_angles_rad



# --- 5. CONTROLADOR LQR ---
class DoubleIntegratorLQRController:
    """ Controlador LQR com modelo de integrador duplo. """
    def __init__(self, dt=0.05):
        self.dt = dt
        self.num_joints = 6
        self.state_dim = 2 * self.num_joints
        
        self.A = np.zeros((self.state_dim, self.state_dim))
        self.B = np.zeros((self.state_dim, self.num_joints))
        for i in range(self.num_joints):
            self.A[i, i] = 1.0; self.A[i, i + self.num_joints] = -dt
            self.A[i + self.num_joints, i + self.num_joints] = 1.0
            self.B[i, i] = -0.5 * dt * dt; self.B[i + self.num_joints, i] = dt
            
        Q = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_joints):
            Q[i, i] = 7000.0; Q[i + self.num_joints, i + self.num_joints] = 30.0
        R = np.eye(self.num_joints) * 0.5
        
        try:
            P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, R)
            self.K = np.linalg.inv(self.B.T @ P @ self.B + R) @ self.B.T @ P @ self.A
            print(f"✓ Ganhos LQR calculados com sucesso!")
        except Exception as e:
            print(f"Erro no cálculo LQR: {e}. Usando ganhos padrão.")
            self.K = np.zeros((self.num_joints, self.state_dim))

        self.reset_state()
        
    def calculate_control(self, current_positions_rad, target_positions_rad):
        position_errors = target_positions_rad - current_positions_rad
        velocities = (current_positions_rad - self.prev_positions_rad) / self.dt if self.prev_positions_rad is not None else np.zeros(self.num_joints)
        self.prev_positions_rad = current_positions_rad.copy()
        
        state = np.concatenate([position_errors, velocities])
        accelerations = -self.K @ state
        velocity_commands = velocities + accelerations * self.dt
        
        MAX_VEL = 20.0
        return np.clip(velocity_commands, -MAX_VEL, MAX_VEL)

    def reset_state(self):
        self.prev_positions_rad = None

def move_to_config_lqr(target_angles_deg, controller, duration_sec=5.0):
    """ Move o robô usando LQR. """
    print(f"\nIniciando movimento LQR para: {target_angles_deg}°")
    target_angles_rad = np.radians(np.array(target_angles_deg))
    dt = controller.dt
    
    time_log, angle_log_rad = [], []
    
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        current_positions_rad = np.array([sim.getJointPosition(h) for h in joint_handles])
        velocity_commands = controller.calculate_control(current_positions_rad, target_angles_rad)
        
        for i in range(6):
            sim.setJointTargetVelocity(joint_handles[i], velocity_commands[i])
        
        time_log.append(time.time() - start_time)
        angle_log_rad.append(current_positions_rad.copy())
        
        time.sleep(dt)
    
    [sim.setJointTargetVelocity(h, 0) for h in joint_handles]
    print("Movimento LQR finalizado.")
    return time_log, angle_log_rad, target_angles_rad

# --- 6. FUNÇÕES DE PLOTAGEM ---
def plot_combined_error_step_response(pid_data, lqr_data):
    """
    Plota o ERRO na resposta ao degrau de AMBOS os controladores.
    """
    all_time_pid, all_angles_pid, all_targets_deg = pid_data
    all_time_lqr, all_angles_lqr, _ = lqr_data

    plt.figure(figsize=(16, 10))
    
    for joint_idx in range(6):
        plt.subplot(3, 2, joint_idx + 1)
        
        # --- Plot ERRO PID ---
        time_offset_pid = 0
        for step_idx, (time_data, angle_history) in enumerate(zip(all_time_pid, all_angles_pid)):
            angle_history_rad = np.array(angle_history)
            target_rad = np.radians(all_targets_deg[step_idx][joint_idx])
            
            # CALCULA O ERRO
            error_rad = target_rad - angle_history_rad[:, joint_idx]
            error_deg = np.degrees(error_rad)
            
            adjusted_time = np.array(time_data) + time_offset_pid
            plt.plot(adjusted_time, error_deg, color='steelblue', linestyle='-', 
                    label='PID' if step_idx == 0 else "")
            time_offset_pid = adjusted_time[-1]

        # --- Plot ERRO LQR ---
        time_offset_lqr = 0
        for step_idx, (time_data, angle_history) in enumerate(zip(all_time_lqr, all_angles_lqr)):
            angle_history_rad = np.array(angle_history)
            target_rad = np.radians(all_targets_deg[step_idx][joint_idx])
            
            # CALCULA O ERRO
            error_rad = target_rad - angle_history_rad[:, joint_idx]
            error_deg = np.degrees(error_rad)
            
            adjusted_time = np.array(time_data) + time_offset_lqr
            plt.plot(adjusted_time, error_deg, color='darkorange', linestyle='-', 
                    label='LQR' if step_idx == 0 else "")
            time_offset_lqr = adjusted_time[-1]

        # --- Linha em zero (referência) ---
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Erro Zero' if joint_idx == 0 else "")
        
        # --- Transições entre degraus ---
        time_offset = 0
        for step_idx in range(len(all_targets_deg)):
            end_time_segment = time_offset + all_time_pid[step_idx][-1]
            if step_idx < len(all_targets_deg) - 1:
                plt.axvline(x=end_time_segment, color='gray', linestyle=':', alpha=0.5)
            time_offset = end_time_segment
            
        plt.title(f'Junta {joint_idx + 1}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Erro (graus)')
        plt.grid(True, alpha=0.4)
        plt.legend()
    
    plt.suptitle('Erro na Resposta ao Degrau: PID vs LQR', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_combined_step_response(pid_data, lqr_data):
    """
    Plota a resposta ao degrau de AMBOS os controladores no mesmo gráfico.
    """
    all_time_pid, all_angles_pid, all_targets_deg = pid_data
    all_time_lqr, all_angles_lqr, _ = lqr_data

    plt.figure(figsize=(16, 10))
    
    for joint_idx in range(6):
        plt.subplot(3, 2, joint_idx + 1)
        
        # --- Plot PID ---
        time_offset_pid = 0
        for step_idx, (time_data, angle_history) in enumerate(zip(all_time_pid, all_angles_pid)):
            angle_history_deg = np.degrees(np.array(angle_history))
            adjusted_time = np.array(time_data) + time_offset_pid
            plt.plot(adjusted_time, angle_history_deg[:, joint_idx], color='steelblue', linestyle='-', label='PID' if step_idx == 0 else "")
            time_offset_pid = adjusted_time[-1]

        # --- Plot LQR ---
        time_offset_lqr = 0
        for step_idx, (time_data, angle_history) in enumerate(zip(all_time_lqr, all_angles_lqr)):
            angle_history_deg = np.degrees(np.array(angle_history))
            adjusted_time = np.array(time_data) + time_offset_lqr
            plt.plot(adjusted_time, angle_history_deg[:, joint_idx], color='darkorange', linestyle='-', label='LQR' if step_idx == 0 else "")
            time_offset_lqr = adjusted_time[-1]

        # --- Plot Alvos (TARGET LINES) e Transições ---
        time_offset = 0
        for step_idx, target_deg in enumerate(all_targets_deg):
            start_time_segment = time_offset
            end_time_segment = time_offset + all_time_pid[step_idx][-1]

            # AQUI ESTÁ A MUDANÇA: Usamos plot() para desenhar um segmento de linha
            plt.plot([start_time_segment, end_time_segment],
                     [target_deg[joint_idx], target_deg[joint_idx]],
                     color='red', linestyle='--', alpha=0.9, label='Alvo' if step_idx == 0 else "")

            if step_idx < len(all_targets_deg) - 1:
                plt.axvline(x=end_time_segment, color='gray', linestyle=':', alpha=0.5)
            
            time_offset = end_time_segment
            
        plt.title(f'Junta {joint_idx + 1}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Ângulo (graus)')
        plt.grid(True, alpha=0.4)
        plt.legend()
    
    plt.suptitle('Comparativo da Resposta ao Degrau: PID vs LQR', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_comparison_summary(metrics_pid, metrics_lqr):
    """ Cria gráfico de barras comparativo entre PID e LQR. """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['Degrau 1', 'Degrau 2', 'Degrau 3', 'Média']
    x = np.arange(len(labels)); width = 0.35
    
    def get_avg_metrics(metrics):
        rmse = [np.degrees(np.mean(m[0])) for m in metrics]
        ts = [np.mean(m[1]) for m in metrics]
        os = [np.mean(m[2]) for m in metrics]
        return rmse + [np.mean(rmse)], ts + [np.mean(ts)], os + [np.mean(os)]

    rmse_pid_avg, ts_pid_avg, os_pid_avg = get_avg_metrics(metrics_pid)
    rmse_lqr_avg, ts_lqr_avg, os_lqr_avg = get_avg_metrics(metrics_lqr)
    
    axes[0].bar(x - width/2, rmse_pid_avg, width, label='PID', color='steelblue', alpha=0.9)
    axes[0].bar(x + width/2, rmse_lqr_avg, width, label='LQR', color='darkorange', alpha=0.9)
    axes[0].set_title('Comparação de RMSE Médio'); axes[0].set_ylabel('RMSE (°)')
    
    axes[1].bar(x - width/2, ts_pid_avg, width, label='PID', color='steelblue', alpha=0.9)
    axes[1].bar(x + width/2, ts_lqr_avg, width, label='LQR', color='darkorange', alpha=0.9)
    axes[1].set_title('Comparação de Tempo de Assentamento Médio'); axes[1].set_ylabel('Tempo (s)')
    
    axes[2].bar(x - width/2, os_pid_avg, width, label='PID', color='steelblue', alpha=0.9)
    axes[2].bar(x + width/2, os_lqr_avg, width, label='LQR', color='darkorange', alpha=0.9)
    axes[2].set_title('Comparação de Overshoot Médio'); axes[2].set_ylabel('Overshoot (%)')

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Análise Comparativa de Desempenho: PID vs LQR', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 7. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    try:
        # --- Configurações Comuns ---
        degrau1_deg = [60, -45, 90, -70, 45, 20]
        degrau2_deg = [-30, 60, -45, 80, -60, 40]
        degrau3_deg = [45, -30, 60, -50, 30, -45]
        todos_os_degraus = [degrau1_deg, degrau2_deg, degrau3_deg]
        duration = 5.0
        
        metrics_pid_all, metrics_lqr_all = [], []
        indices_pid_all, indices_lqr_all = [], []  # NOVO: para armazenar índices
        
        # ======================================================
        #  BLOCO DE TESTE PID
        # ======================================================
        print("\n" + "#"*30 + " INICIANDO TESTE PID " + "#"*30)
        sim.stopSimulation(True); time.sleep(1); sim.startSimulation(); time.sleep(1)
        
        all_time_pid, all_angles_pid, all_targets_pid_deg = [], [], []
        
        for i, degrau_deg in enumerate(todos_os_degraus):
            time_data, angle_hist_rad, target_rad = move_to_config_pid(degrau_deg, duration_sec=duration)
            
            # Métricas originais
            rmse, ts, overshoot = calculate_metrics(time_data, angle_hist_rad, target_rad)
            print_metrics_summary(rmse, ts, overshoot, f"PID - DEGRAU {i+1}")
            metrics_pid_all.append((rmse, ts, overshoot))
            
            # NOVO: Calcula índices de desempenho
            indices = calcular_indices_desempenho(time_data, angle_hist_rad, target_rad)
            print_indices_summary(indices, f"PID - DEGRAU {i+1}")
            indices_pid_all.append(indices)
            
            all_time_pid.append(time_data)
            all_angles_pid.append(angle_hist_rad)
            all_targets_pid_deg.append(degrau_deg)
            time.sleep(0.5)
            
        # ======================================================
        #  BLOCO DE TESTE LQR
        # ======================================================
        print("\n" + "#"*30 + " INICIANDO TESTE LQR " + "#"*30)
        sim.stopSimulation(True); time.sleep(1); sim.startSimulation(); time.sleep(1)
        
        controller_lqr = DoubleIntegratorLQRController(dt=0.05)
        all_time_lqr, all_angles_lqr, all_targets_lqr_deg = [], [], []

        for i, degrau_deg in enumerate(todos_os_degraus):
            controller_lqr.reset_state()
            time_data, angle_hist_rad, target_rad = move_to_config_lqr(degrau_deg, controller_lqr, duration_sec=duration)
            
            # Métricas originais
            rmse, ts, overshoot = calculate_metrics(time_data, angle_hist_rad, target_rad)
            print_metrics_summary(rmse, ts, overshoot, f"LQR - DEGRAU {i+1}")
            metrics_lqr_all.append((rmse, ts, overshoot))

            # NOVO: Calcula índices de desempenho
            indices = calcular_indices_desempenho(time_data, angle_hist_rad, target_rad)
            print_indices_summary(indices, f"LQR - DEGRAU {i+1}")
            indices_lqr_all.append(indices)

            all_time_lqr.append(time_data)
            all_angles_lqr.append(angle_hist_rad)
            all_targets_lqr_deg.append(degrau_deg)
            time.sleep(0.5)

        # ======================================================
        #  RESUMO COMPARATIVO DOS ÍNDICES
        # ======================================================
        print("\n" + "="*70)
        print("RESUMO COMPARATIVO: ÍNDICES DE DESEMPENHO MÉDIOS")
        print("="*70)
        
        # Calcula médias gerais
        iae_pid_media = np.mean([idx['IAE'] for idx in indices_pid_all])
        ise_pid_media = np.mean([idx['ISE'] for idx in indices_pid_all])
        itae_pid_media = np.mean([idx['ITAE'] for idx in indices_pid_all])
        itse_pid_media = np.mean([idx['ITSE'] for idx in indices_pid_all])
        
        iae_lqr_media = np.mean([idx['IAE'] for idx in indices_lqr_all])
        ise_lqr_media = np.mean([idx['ISE'] for idx in indices_lqr_all])
        itae_lqr_media = np.mean([idx['ITAE'] for idx in indices_lqr_all])
        itse_lqr_media = np.mean([idx['ITSE'] for idx in indices_lqr_all])
        print(f"\n{'Controlador':<15} {'IAE':<15} {'ISE':<15} {'ITAE':<15} {'ITSE':<15}")
        print("-"*70)
        print(f"{'PID':<15} {iae_pid_media:<15.4f} {ise_pid_media:<15.4f} "
              f"{itae_pid_media:<15.4f} {itse_pid_media:<15.4f}")
        print(f"{'LQR':<15} {iae_lqr_media:<15.4f} {ise_lqr_media:<15.4f} "
              f"{itae_lqr_media:<15.4f} {itse_lqr_media:<15.4f}")
        print("-"*70)
        
        # Calcula melhoria percentual
        melhoria_iae = ((iae_pid_media - iae_lqr_media) / iae_pid_media) * 100
        melhoria_ise = ((ise_pid_media - ise_lqr_media) / ise_pid_media) * 100
        melhoria_itae = ((itae_pid_media - itae_lqr_media) / itae_pid_media) * 100
        melhoria_itse = ((itse_pid_media - itse_lqr_media) / itse_pid_media) * 100
        
        print(f"{'Melhoria (%)':<15} {melhoria_iae:<15.2f} {melhoria_ise:<15.2f} "
              f"{melhoria_itae:<15.2f} {melhoria_itse:<15.2f}")
        print("="*70)
        
        # ======================================================
        #  BLOCO DE PLOTAGEM E COMPARAÇÃO FINAL
        # ======================================================
        print("\n" + "#"*30 + " GERANDO GRÁFICOS COMPARATIVOS " + "#"*30)
        
        # Agrupa os dados para a nova função de plotagem
        pid_plot_data = (all_time_pid, all_angles_pid, all_targets_pid_deg)
        lqr_plot_data = (all_time_lqr, all_angles_lqr, all_targets_lqr_deg)

        # Plota o gráfico de ERRO primeiro
        plot_combined_error_step_response(pid_plot_data, lqr_plot_data)
        
        print("\nGerando gráfico de erro euclidiano para resposta ao degrau...")
        plot_erro_cartesiano_degrau(pid_plot_data, lqr_plot_data)

        # Plota o gráfico de posição (original)
        plot_combined_step_response(pid_plot_data, lqr_plot_data)

        # Chama a função que plota o resumo das métricas em barras
        plot_comparison_summary(metrics_pid_all, metrics_lqr_all)

    except Exception as e:
        print(f"\nOcorreu um erro na execução principal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExecutando limpeza final...")
        try:
            sim.stopSimulation(True)
            print("Simulação parada.")
        except:
            pass
        plt.close('all')
        print("Programa finalizado.")
