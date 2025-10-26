import numpy as np
import math
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg

# --- 1. INICIALIZAÇÃO E HANDLES ---
try:
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    joint_handles = [sim.getObject(f'/UR10/joint{i+1}') for i in range(6)]
    tip_handle = sim.getObject('/UR10/tip')
    print("Conectado e handles obtidos.")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível inicializar. Erro: {e}")
    exit()

# --- 2. FUNÇÕES DE TRAJETÓRIA E MOVIMENTO ---
def gerar_trajetoria_senoidal(t_final, dt, juntas_alvo, angulos_iniciais_rad):
    n_passos = int(t_final / dt)
    tempo = np.linspace(0, t_final, n_passos)
    trajetoria = np.tile(angulos_iniciais_rad, (n_passos, 1))
    for i, params in juntas_alvo.items():
        amp = np.radians(params['amp']); freq = params['freq']; offset = np.radians(params['offset'])
        trajetoria[:, i] = offset + amp * np.sin(2 * np.pi * freq * tempo)
    return tempo, trajetoria

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def mover_para_ponto_inicial(ponto_alvo_rad, dt):
    print("\nMovendo para a posição inicial da trajetória...")
    Kp = 15.0; Ki = 5.0; Kd = 0.1
    integral_error = np.zeros(6); previous_error = np.zeros(6)
    for _ in range(200):
        current_angles = np.array([sim.getJointPosition(h) for h in joint_handles])
        error = normalize_angle(ponto_alvo_rad - current_angles)
        if np.linalg.norm(np.degrees(error)) < 0.5:
            print("Posição inicial alcançada.")
            [sim.setJointTargetVelocity(h, 0) for h in joint_handles]; time.sleep(0.5)
            return
        integral_error += error * dt; derivative_error = (error - previous_error) / dt
        target_velocities = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
        for i in range(6): sim.setJointTargetVelocity(joint_handles[i], target_velocities[i])
        previous_error = error; time.sleep(dt)
    print("Não foi possível alcançar a posição inicial com precisão. Continuando...")
    [sim.setJointTargetVelocity(h, 0) for h in joint_handles]; time.sleep(0.5)

# --- 3. FUNÇÕES DE CONTROLE DE TRAJETÓRIA ---
def seguir_trajetoria_pid(trajetoria_rad, dt):
    print("Iniciando seguimento de trajetória com PID...")
    Kp, Ki, Kd = 8.0, 1.5, 0.5
    integral_error = np.zeros(6); previous_error = np.zeros(6)
    angle_log = []; tip_pos_log = []
    for k in range(len(trajetoria_rad)):
        target_angles_rad = trajetoria_rad[k]
        current_angles = np.array([sim.getJointPosition(h) for h in joint_handles])
        error = normalize_angle(target_angles_rad - current_angles)
        integral_error += error * dt; derivative_error = (error - previous_error) / dt
        target_velocities = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
        for i in range(6): sim.setJointTargetVelocity(joint_handles[i], target_velocities[i])
        tip_pos_log.append(sim.getObjectPosition(tip_handle, -1))
        angle_log.append(current_angles); previous_error = error; time.sleep(dt)
    [sim.setJointTargetVelocity(h, 0) for h in joint_handles]
    print("Seguimento de trajetória PID finalizado.")
    return np.array(angle_log), np.array(tip_pos_log)

class DoubleIntegratorLQRController:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.num_joints = 6
        self.state_dim = 2 * self.num_joints
        self.A = np.zeros((self.state_dim, self.state_dim))
        self.B = np.zeros((self.state_dim, self.num_joints))
        for i in range(self.num_joints):
            self.A[i, i] = 1.0
            self.A[i, i + self.num_joints] = -dt
            self.A[i + self.num_joints, i + self.num_joints] = 1.0
            self.B[i, i] = -0.5 * dt * dt
            self.B[i + self.num_joints, i] = dt
        Q = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_joints):
            Q[i, i] = 50000.0
            Q[i + self.num_joints, i + self.num_joints] = 30.0
        R = np.eye(self.num_joints) * 0.5
        self.P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, R)
        BT_P_B_plus_R = self.B.T @ self.P @ self.B + R
        self.K = np.linalg.inv(BT_P_B_plus_R) @ self.B.T @ self.P @ self.A
        self.reset_state()

    def reset_state(self):
        self.prev_positions = None
       
    def calculate_control(self, state, target_velocities):
        accelerations = -self.K @ state
        velocity_commands = target_velocities + accelerations * self.dt
        return np.clip(velocity_commands, -20.0, 20.0)

    def estimate_state(self, current_positions, target_positions, target_velocities):
        position_errors = normalize_angle(target_positions - current_positions)
        
        # CORREÇÃO APLICADA AQUI
        if self.prev_positions is not None:
            current_velocities = (current_positions - self.prev_positions) / self.dt
        else:
            # Na primeira iteração, assume velocidade atual = velocidade desejada
            current_velocities = target_velocities
        
        self.prev_positions = current_positions.copy()

        velocity_errors = current_velocities - target_velocities
        
        state = np.concatenate([position_errors, velocity_errors])
        
        return state, current_velocities

def seguir_trajetoria_lqr(trajetoria_rad, dt):
    print("Iniciando seguimento de trajetória com LQR (Corrigido)...")
    controller = DoubleIntegratorLQRController(dt)
    angle_log = []; tip_pos_log = []

    velocidades_desejadas_rad = np.gradient(trajetoria_rad, dt, axis=0)
    
    # NOVO: Período de aquecimento (primeiros 0.5s com ganhos reduzidos)
    warmup_steps = int(0.5 / dt)  # 0.5 segundos de aquecimento
    
    for k in range(len(trajetoria_rad)):
        target_angles_rad = trajetoria_rad[k]
        target_velocities_rad = velocidades_desejadas_rad[k]
        
        current_angles = np.array([sim.getJointPosition(h) for h in joint_handles])
        
        state, current_velocities = controller.estimate_state(
            current_angles, target_angles_rad, target_velocities_rad
        )
        
        velocity_commands = controller.calculate_control(state, target_velocities_rad)
        
        # Durante o aquecimento, aplica fator de suavização
        if k < warmup_steps:
            alpha = k / warmup_steps  # Vai de 0 a 1
            velocity_commands = velocity_commands * alpha
        
        for i in range(6): 
            sim.setJointTargetVelocity(joint_handles[i], velocity_commands[i])
        
        tip_pos_log.append(sim.getObjectPosition(tip_handle, -1))
        angle_log.append(current_angles)
        time.sleep(dt)
        
    [sim.setJointTargetVelocity(h, 0) for h in joint_handles]
    print("Seguimento de trajetória LQR finalizado.")
    return np.array(angle_log), np.array(tip_pos_log)
    
# --- 4. FUNÇÕES DE PLOTAGEM ---
def calcular_caminho_3d_desejado(trajetoria_rad):
    print("Pré-calculando caminho 3D desejado...")
    caminho_3d = []
    sim.pauseSimulation()
    try:
        posicao_original = np.array([sim.getJointPosition(h) for h in joint_handles])
        for angulos in trajetoria_rad:
            for i in range(6): sim.setJointPosition(joint_handles[i], angulos[i])
            caminho_3d.append(sim.getObjectPosition(tip_handle, -1))
        for i in range(6): sim.setJointPosition(joint_handles[i], posicao_original[i])
    finally:
        sim.startSimulation()
    print("Cálculo finalizado com sucesso.")
    return np.array(caminho_3d)

def plot_trajetoria_comparativo_2d(tempo, trajetoria_desejada_rad, pid_real_rad, lqr_real_rad):
    plt.figure(figsize=(16, 10)); desejada_deg = np.degrees(trajetoria_desejada_rad)
    pid_deg = np.degrees(pid_real_rad); lqr_deg = np.degrees(lqr_real_rad)
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(tempo, desejada_deg[:, i], color='black', linestyle='--', label='Desejada')
        plt.plot(tempo, pid_deg[:, i], color='steelblue', alpha=0.9, label='PID Real')
        plt.plot(tempo, lqr_deg[:, i], color='darkorange', alpha=0.9, label='LQR Real')
        plt.title(f'Junta {i + 1}'); plt.xlabel('Tempo (s)'); plt.ylabel('Ângulo (°)')
        plt.grid(True, alpha=0.5); plt.legend()
    plt.suptitle('Comparativo de Seguimento de Trajetória (Ângulos de Junta)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def plot_erro_trajetoria_comparativo(tempo, trajetoria_desejada_rad, pid_real_rad, lqr_real_rad):
    """
    Plota o ERRO de seguimento de trajetória para cada junta.
    """
    plt.figure(figsize=(16, 10))
    erro_pid_rad = trajetoria_desejada_rad - pid_real_rad
    erro_lqr_rad = trajetoria_desejada_rad - lqr_real_rad
    erro_pid_deg = np.degrees(erro_pid_rad)
    erro_lqr_deg = np.degrees(erro_lqr_rad)
    
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(tempo, erro_pid_deg[:, i], color='steelblue', alpha=0.9, label='Erro PID')
        plt.plot(tempo, erro_lqr_deg[:, i], color='darkorange', alpha=0.9, label='Erro LQR')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Erro Zero')
        plt.title(f'Junta {i + 1}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Erro (°)')
        plt.grid(True, alpha=0.5)
        plt.legend()
    
    plt.suptitle('Erro no Seguimento de Trajetória: PID vs LQR', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_erro_cartesiano(tempo, pos_desejada, pos_pid, pos_lqr):
    """
    Plota APENAS o erro Euclidiano total, com zoom ajustado para focar na tendência.
    """
    # Define o tempo de corte para ignorar o pico inicial
    tempo_corte = 0.5
    dt = tempo[1] - tempo[0]
    start_index = int(tempo_corte / dt)

    # Fatiamento dos dados para remover o transiente
    tempo_plot = tempo[start_index:]
    pos_desejada_plot = pos_desejada[start_index:]
    pos_pid_plot = pos_pid[start_index:]
    pos_lqr_plot = pos_lqr[start_index:]

    # Calcula erros vetoriais
    erro_pid = pos_desejada_plot - pos_pid_plot
    erro_lqr = pos_desejada_plot - pos_lqr_plot
    
    # Calcula a norma Euclidiana (erro total) e converte para mm
    erro_norm_pid = np.linalg.norm(erro_pid, axis=1) * 1000
    erro_norm_lqr = np.linalg.norm(erro_lqr, axis=1) * 1000
    
    # --- GRÁFICO ÚNICO E FOCADO ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(tempo_plot, erro_norm_pid, 'steelblue', linewidth=2.2, label='PID', alpha=0.9)
    ax.plot(tempo_plot, erro_norm_lqr, 'darkorange', linewidth=2.2, label='LQR', alpha=0.9)
    
    ax.set_title('Erro Euclidiano Total no Seguimento de Trajetória', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erro Euclidiano Total (mm)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # --- AJUSTE DE ZOOM PARA "ESCONDER" RUÍDO ---
    # Encontra o maior pico de erro para definir o limite do eixo Y
    max_error = max(np.max(erro_norm_pid), np.max(erro_norm_lqr))
    ax.set_ylim(0, max_error * 1.15)  # Define o limite Y com 15% de margem superior
    
    plt.tight_layout()
    plt.show()
    
    # --- TABELA DE MÉTRICAS (ESSENCIAL PARA A ANÁLISE) ---
    print("\n" + "="*70)
    print("ESTATÍSTICAS DO ERRO CARTESIANO TOTAL (IGNORANDO TRANSIENTE INICIAL)")
    print("="*70)
    print(f"{'Controlador':<15} {'Erro Médio (mm)':<20} {'Erro Máximo (mm)':<20} {'Erro RMS (mm)':<20}")
    print("-"*70)
    print(f"{'PID':<15} {np.mean(erro_norm_pid):<20.4f} {np.max(erro_norm_pid):<20.4f} {np.sqrt(np.mean(erro_norm_pid**2)):<20.4f}")
    print(f"{'LQR':<15} {np.mean(erro_norm_lqr):<20.4f} {np.max(erro_norm_lqr):<20.4f} {np.sqrt(np.mean(erro_norm_lqr**2)):<20.4f}")
    print("="*70)

def plot_caminho_3d_comparativo(desejado_pos, pid_pos, lqr_pos):
    fig = plt.figure(figsize=(12, 9)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(desejado_pos[:, 0], desejado_pos[:, 1], desejado_pos[:, 2], 
               marker='.', color='gray', alpha=0.4, label='Caminho Desejado')
    ax.plot(pid_pos[:, 0], pid_pos[:, 1], pid_pos[:, 2], color='steelblue', linewidth=2, label='Caminho PID')
    ax.plot(lqr_pos[:, 0], lqr_pos[:, 1], lqr_pos[:, 2], color='darkorange', linewidth=2.5, label='Caminho LQR')
    ax.scatter(lqr_pos[0,0], lqr_pos[0,1], lqr_pos[0,2], color='g', s=150, label='Início', depthshade=False, edgecolors='black')
    ax.scatter(lqr_pos[-1,0], lqr_pos[-1,1], lqr_pos[-1,2], color='r', s=150, label='Fim', depthshade=False, edgecolors='black')
    ax.set_title('Caminho 3D do Efetuador: Desejado vs. PID vs. LQR', fontsize=16, pad=20)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(); ax.grid(True)
    max_range = np.array([desejado_pos[:,0].max()-desejado_pos[:,0].min(), desejado_pos[:,1].max()-desejado_pos[:,1].min(), desejado_pos[:,2].max()-desejado_pos[:,2].min()]).max()/1.5
    mid_x, mid_y, mid_z = (desejado_pos.max(axis=0)+desejado_pos.min(axis=0))*0.5
    ax.set_xlim(mid_x-max_range, mid_x+max_range); ax.set_ylim(mid_y-max_range, mid_y+max_range); ax.set_zlim(mid_z-max_range, mid_z+max_range)
    plt.show()

def calcular_metricas_trajetoria(nome_controlador, tempo, desejada_rad, real_rad):
    """
    Calcula e imprime o RMSE e o Erro Máximo para cada junta.
    """
    print(f"\n--- Métricas de Desempenho para: {nome_controlador} ---")
    desejada_deg = np.degrees(desejada_rad)
    real_deg = np.degrees(real_rad)
    erro_deg = desejada_deg - real_deg
    rmse_por_junta = np.sqrt(np.mean(erro_deg**2, axis=0))
    max_erro_por_junta = np.max(np.abs(erro_deg), axis=0)
    
    print(f"{'Junta':<10} {'RMSE (°)' :<15} {'Erro Máximo (°)' :<20}")
    print("-" * 50)
    
    for i in range(6):
        print(f"Junta {i+1:<5} {rmse_por_junta[i]:<15.4f} {max_erro_por_junta[i]:<20.4f}")
        
    print("-" * 50)
    rmse_geral = np.sqrt(np.mean(erro_deg**2))
    print(f"{'MÉDIA GERAL RMSE:':<25} {rmse_geral:.4f}°")
    print("-" * 50)

def calcular_indices_desempenho_trajetoria(tempo, desejada_rad, real_rad):
    """
    Calcula os índices de desempenho IAE, ISE, ITAE, ITSE para seguimento de trajetória.
    """
    tempo = np.array(tempo)
    desejada_rad = np.array(desejada_rad)
    real_rad = np.array(real_rad)
    n_joints = desejada_rad.shape[1]
    
    IAE_joints = np.zeros(n_joints)
    ISE_joints = np.zeros(n_joints)
    ITAE_joints = np.zeros(n_joints)
    ITSE_joints = np.zeros(n_joints)
    
    for i in range(n_joints):
        error = desejada_rad[:, i] - real_rad[:, i]
        error_abs = np.abs(error)
        error_squared = error ** 2
        IAE_joints[i] = np.trapz(error_abs, tempo)
        ISE_joints[i] = np.trapz(error_squared, tempo)
        ITAE_joints[i] = np.trapz(tempo * error_abs, tempo)
        ITSE_joints[i] = np.trapz(tempo * error_squared, tempo)
    
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

def print_indices_trajetoria(indices, nome_controlador):
    """
    Imprime os índices de desempenho de trajetória de forma organizada.
    """
    print(f"\n{'='*70}")
    print(f"ÍNDICES DE DESEMPENHO - {nome_controlador}")
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

# --- 5. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    try:
        # --- Configurações da Trajetória ---
        dt = 0.02; duracao = 10.0
        pos_inicial_deg = np.array([0, 0, 90, 0, 90, 0])
        alvos_senoidais = {
            0: {'amp': 45, 'freq': 0.2, 'offset': pos_inicial_deg[0]},
            2: {'amp': 30, 'freq': 0.3, 'offset': pos_inicial_deg[2]},
            4: {'amp': 40, 'freq': 0.25, 'offset': pos_inicial_deg[4]}
        }
        
        # --- Pré-Cálculos ---
        sim.stopSimulation(True); time.sleep(1); sim.startSimulation(); time.sleep(1)
        tempo, trajetoria_rad = gerar_trajetoria_senoidal(duracao, dt, alvos_senoidais, np.radians(pos_inicial_deg))
        ponto_de_partida_rad = trajetoria_rad[0]
        caminho_3d_desejado = calcular_caminho_3d_desejado(trajetoria_rad)
        sim.stopSimulation(True); time.sleep(1)
        
        # --- Teste PID ---
        print("\n" + "#"*30 + " INICIANDO TESTE DE TRAJETÓRIA PID " + "#"*30)
        sim.startSimulation(); time.sleep(1)
        mover_para_ponto_inicial(ponto_de_partida_rad, dt)
        pid_real_rad, pid_tip_pos = seguir_trajetoria_pid(trajetoria_rad, dt)
        sim.stopSimulation(True); time.sleep(1)  
              
        # --- Teste LQR ---
        print("\n" + "#"*30 + " INICIANDO TESTE DE TRAJETÓRIA LQR " + "#"*30)
        sim.startSimulation(); time.sleep(1)
        mover_para_ponto_inicial(ponto_de_partida_rad, dt)
        lqr_real_rad, lqr_tip_pos = seguir_trajetoria_lqr(trajetoria_rad, dt)
        sim.stopSimulation(True); time.sleep(1)


        
        # --- Métricas Originais (RMSE) ---
        calcular_metricas_trajetoria("PID", tempo, trajetoria_rad, pid_real_rad)
        calcular_metricas_trajetoria("LQR", tempo, trajetoria_rad, lqr_real_rad)
        
        # --- Índices de Desempenho ---
        print("\n" + "#"*30 + " CALCULANDO ÍNDICES DE DESEMPENHO " + "#"*30)
        indices_pid = calcular_indices_desempenho_trajetoria(tempo, trajetoria_rad, pid_real_rad)
        print_indices_trajetoria(indices_pid, "PID")
        indices_lqr = calcular_indices_desempenho_trajetoria(tempo, trajetoria_rad, lqr_real_rad)
        print_indices_trajetoria(indices_lqr, "LQR")
        
        # --- Resumo Comparativo ---
        print("\n" + "="*70)
        print("RESUMO COMPARATIVO: ÍNDICES DE DESEMPENHO")
        print("="*70)
        print(f"\n{'Controlador':<15} {'IAE':<15} {'ISE':<15} {'ITAE':<15} {'ITSE':<15}")
        print("-"*70)
        print(f"{'PID':<15} {indices_pid['IAE']:<15.4f} {indices_pid['ISE']:<15.4f} "
              f"{indices_pid['ITAE']:<15.4f} {indices_pid['ITSE']:<15.4f}")
        print(f"{'LQR':<15} {indices_lqr['IAE']:<15.4f} {indices_lqr['ISE']:<15.4f} "
              f"{indices_lqr['ITAE']:<15.4f} {indices_lqr['ITSE']:<15.4f}")
        print("-"*70)
        
        melhoria_iae = ((indices_pid['IAE'] - indices_lqr['IAE']) / indices_pid['IAE']) * 100
        melhoria_ise = ((indices_pid['ISE'] - indices_lqr['ISE']) / indices_pid['ISE']) * 100
        melhoria_itae = ((indices_pid['ITAE'] - indices_lqr['ITAE']) / indices_pid['ITAE']) * 100
        melhoria_itse = ((indices_pid['ITSE'] - indices_lqr['ITSE']) / indices_pid['ITSE']) * 100
        
        print(f"{'Melhoria (%)':<15} {melhoria_iae:<15.2f} {melhoria_ise:<15.2f} "
              f"{melhoria_itae:<15.2f} {melhoria_itse:<15.2f}")
        print("="*70)
		
        # --- Plotagem Final ---
        print("\n" + "#"*30 + " GERANDO GRÁFICOS COMPARATIVOS " + "#"*30)
        plot_trajetoria_comparativo_2d(tempo, trajetoria_rad, pid_real_rad, lqr_real_rad)
        plot_erro_trajetoria_comparativo(tempo, trajetoria_rad, pid_real_rad, lqr_real_rad)
        
        # --- NOVO: Gráfico de Erro Cartesiano (MAIS LIMPO E PROFISSIONAL) ---
        print("\nGerando gráfico de erro cartesiano...")
        plot_erro_cartesiano(tempo, caminho_3d_desejado, pid_tip_pos, lqr_tip_pos)
        
        plot_caminho_3d_comparativo(caminho_3d_desejado, pid_tip_pos, lqr_tip_pos)

    except Exception as e:
        print(f"\nOcorreu um erro na execução principal: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: sim.stopSimulation(True); print("Simulação parada.")
        except: pass
