from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import random
import cv2
import math

# ==============================
# Inicialização
# ==============================
client = RemoteAPIClient()
sim = client.getObject('sim')

# Handles principais
goalPose = sim.getObject('/goalPose')
camera = sim.getObject('/VisionSensorOrtho')
DISPLAY_SIZE = 400
# ==============================
# Funções auxiliares
# ==============================

def start_sim():
    sim.stopSimulation()
    time.sleep(0.5) # Pausa para garantir que parou
    sim.startSimulation()
    time.sleep(1) # Pausa para os scripts LUA inicializarem
    print("Simulação iniciada")

def stop_sim():
    sim.stopSimulation()
    print("Simulação parada")


# ------------------------------
# Controle da garra (RG2) - VERSÃO MELHORADA
# ------------------------------
def set_gripper(state, wait_time=1.5):
    """
    Define o estado da garra.
    state: True para ABRIR, False para FECHAR.
    """
    signal_value = 1 if state else 0
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', signal_value)
    action = "aberta" if state else "fechada"
    print(f"Garra comandada para ser {action}")
    time.sleep(wait_time)

# -------------------------------
# Função SLERP para interpolar quaternions
# -------------------------------
def slerp(q1, q2, t):
    """Slerp simples entre q1 e q2"""
    dot = sum(a*b for a,b in zip(q1,q2))

    # Corrige se o produto escalar for negativo (escolhe o menor arco)
    if dot < 0.0:
        q2 = [-v for v in q2]
        dot = -dot

    if dot > 0.9995:
        # Quase iguais → interp linear normalizada
        result = [a + t*(b-a) for a,b in zip(q1,q2)]
        norm = math.sqrt(sum(r*r for r in result))
        return [r/norm for r in result]

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return [s0*a + s1*b for a,b in zip(q1,q2)]


# ------------------------------
# Movimento do braço via IK
# ------------------------------
def euler_to_quat(roll, pitch, yaw):
    """Converte ângulos Euler (rad) em quaternion [x,y,z,w]"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return [qx, qy, qz, qw]

def move_to_pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0, steps=20):
    """Move suavemente até (x,y,z,roll,pitch,yaw)"""
    start_pose = sim.getObjectPose(goalPose, -1)

    start_pos = start_pose[0:3]
    start_quat = start_pose[3:7]

    target_pos = [x, y, z]
    target_quat = euler_to_quat(math.radians(roll), math.radians(pitch), math.radians(yaw))

    for i in range(1, steps+1):
        t = i/steps
        # Interpola posição
        pos = [(1-t)*a + t*b for a,b in zip(start_pos, target_pos)]
        # Interpola orientação
        quat = slerp(start_quat, target_quat, t)

        pose = pos + quat
        sim.setObjectPose(goalPose, -1, pose)
        cv2.waitKey(15)        
  
# ------------------------------
# Spawn de blocos
# ------------------------------
def spawn_block(color=None):
    """Spawna um bloco com cor (RGB) aleatória DENTRO DE UMA ÁREA SEGURA."""
    # Limites originais da área de spawn
    xmin_orig=0.61
    xmax_orig=1.10
    ymin_orig=-0.45
    ymax_orig=0.45

    # Margem de segurança para garantir que o bloco nunca apareça no canto
    margem = 0.075

    # Novos limites com a margem aplicada
    xmin = xmin_orig + margem
    xmax = xmax_orig - margem
    ymin = ymin_orig + margem
    ymax = ymax_orig - margem

    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    z = 0.025

    # Criar cubo
    block = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.05, 0.05, 0.05], 2)
    
    # Forçar respondable = True
    sim.setObjectInt32Param(block, sim.shapeintparam_respondable, 1)
    # Forçar static = False (para ser dinâmico)
    sim.setObjectInt32Param(block, sim.shapeintparam_static, 0)
    # Verificar se as propriedades foram aplicadas
    is_respondable = sim.getObjectInt32Param(block, sim.shapeintparam_respondable)
    is_static = sim.getObjectInt32Param(block, sim.shapeintparam_static)

    ang = random.uniform(-180.0, 180.0)
    rot = math.radians(ang)
    sim.setObjectPose(block, -1, [x, y, z, 0, 0, math.sin(rot/2.0), math.cos(rot/2.0)])
    
    # Definir cor
    #color_option = [[0, 255 , 0] , [0, 255, 0], [0, 255, 0]]
    color_option = [[255, 0 , 0] , [0, 255, 0], [0, 0, 255]]
    color = random.choice(color_option)
    sim.setShapeColor(block, None, sim.colorcomponent_ambient_diffuse, [c/255.0 for c in color])
    
    print(f"Bloco spawnado em ({x:.2f}, {y:.2f}, yaw={ang:.1f}°), cor={color}")
    return block
    
    
def despawn_block(block_handle):
    """Despawn bloco ao ser depositado"""
    sim.removeObject(block_handle)

# ------------------------------
# Processamento de visão
# ------------------------------
def get_camera_image():
    img, res = sim.getVisionSensorImg(camera)
    img = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def detect_block(img):
    """
    Detecta cor do bloco na imagem e retorna centro em pixel e cor.
    Para cores puras (255,0,0), (0,255,0), (0,0,255) em ambiente simulado.
    """
    if img is None:
        return None, None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definições para cores puras em simulação
    color_ranges = {
        "red":   ([0, 200, 200], [10, 255, 255]),     # Vermelho puro
        "green": ([50, 200, 200], [70, 255, 255]),    # Verde puro  
        "blue":  ([110, 200, 200], [130, 255, 255])   # Azul puro
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Só precisa de um filtro básico
        M = cv2.moments(mask)
        if M["m00"] > 100:  # área mínima pequena para filtrar ruído
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), color_name
    
    return None, None
    
    
def estimate_block_orientation(img, color="red"):
    """
    Estima orientação do bloco baseado na cor especificada.
    Retorna ângulo em radianos.
    """
    if img is None:
        return None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mesmas faixas da função detect_block (para cores puras)
    color_ranges = {
        "red":   ([0, 200, 200], [10, 255, 255]),
        "green": ([50, 200, 200], [70, 255, 255]),
        "blue":  ([110, 200, 200], [130, 255, 255])
    }
    
    if color not in color_ranges:
        return None
        
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Pegar o maior contorno
    c = max(contours, key=cv2.contourArea)
    
    # Verificar se o contorno é grande o suficiente
    if cv2.contourArea(c) < 100:
        return None
    
    # Calcular retângulo mínimo
    rect = cv2.minAreaRect(c)
    center, (width, height), angle = rect
    
    # minAreaRect retorna ângulo entre -90 e 0
    # Ajustar para que 0° seja horizontal
    if width < height:
        angle += 90
    
    # Converter para radianos
    yaw_rad = math.radians(angle)
    
    # Normalizar para range [-π, π]
    while yaw_rad > math.pi:
        yaw_rad -= 2 * math.pi
    while yaw_rad < -math.pi:
        yaw_rad += 2 * math.pi
    
    return yaw_rad

# ------------------------------
# Converter pixel → coordenada mundo (PARA CÂMERA ORTOGONAL)
# ------------------------------
def pixel_to_world_ortho(cx, cy):
    """
    Converte pixel (cx, cy) em coordenada real (x,y,z) para uma câmera ortogonal.
    """
    # Parâmetros exatos da sua câmera no CoppeliaSim
    camera_pos = (0.9, 0.0, 0.5)
    img_res = (1024, 1024)
    ortho_size = 0.80 # Valor da sua configuração

    # A orientação da câmera continua a mesma (eixos trocados/invertidos)
    # alpha=-180, beta=0, gamma=-90

    width, height = img_res
    camera_x, camera_y, _ = camera_pos

    # Para uma câmera ortogonal, a área de visão é um retângulo.
    # Como a resolução é quadrada (1024x1024), a largura e altura da visão são iguais ao ortho_size.
    view_size = ortho_size

    # Normaliza as coordenadas do pixel para o intervalo [-0.5, +0.5]
    nx = (cx / width) - 0.5
    ny = 0.5 - (cy / height) 

    # --- A LÓGICA DE MAPEAMENTO LINEAR ---
    # Mapeia os eixos da câmera para os eixos do mundo com base na orientação
    # O movimento vertical na imagem (ny) muda a coordenada X do mundo (invertido).
    # O movimento horizontal na imagem (nx) muda a coordenada Y do mundo (invertido).
    x_world = camera_x - (ny * view_size)
    y_world = camera_y - (nx * view_size)
    z_world = 0.025  # Altura para o centro do bloco

    return x_world, y_world, z_world
    
    
# ------------------------------
# Converter pixel → coordenada mundo (Perspective)
# ------------------------------
"""
def pixel_to_world_persp(cx, cy):
    
    Converte pixel (cx, cy) em coordenada real (x,y,z) no plano Z=0,
    usando os parâmetros REAIS da câmera na cena do CoppeliaSim.
    
    # Parâmetros exatos da sua câmera no CoppeliaSim
    camera_pos = (0.9, 0.0, 0.5)
    img_res = (1024, 1024)
    persp_angle_deg = 80.0

    # Orientação da câmera: alpha=-180, beta=0, gamma=-90
    # Isso significa que o eixo X da câmera aponta para o Y do mundo,
    # e o eixo Y da câmera aponta para o -X do mundo.

    width, height = img_res
    camera_x, camera_y, camera_z = camera_pos

    # Distância da câmera até o plano dos blocos (Z=0)
    dist_z = camera_z

    # Ângulo de visão em radianos
    angle_rad = math.radians(persp_angle_deg)

    # Largura da área visível no plano Z=0
    view_width_at_z0 = 2 * dist_z * math.tan(angle_rad / 2)

    # Como a resolução é quadrada, a altura visível é a mesma
    view_height_at_z0 = view_width_at_z0

    # Normaliza as coordenadas do pixel para o intervalo [-0.5, +0.5]
    # (0,0) no centro da imagem
    nx = (cx / width) - 0.5
    ny = 0.5 - (cy / height) 

    # --- A GRANDE CORREÇÃO ESTÁ AQUI ---
    # Mapeia os eixos da câmera para os eixos do mundo com base na orientação
    # O movimento horizontal na imagem (nx) muda a coordenada Y do mundo.
    # O movimento vertical na imagem (ny) muda a coordenada X do mundo (invertido).
    x_world = camera_x - (ny * view_height_at_z0)
    y_world = camera_y - (nx * view_width_at_z0)
    z_world = 0.025  # Altura para o centro do bloco

    return x_world, y_world, z_world
"""

def detect_block_world():
    """
    Função completa:
    - Captura a imagem
    - Detecta bloco
    - Calcula posição em mundo
    - Estima yaw
    Retorna: x, y, z, yaw, cor
    """
    img = get_camera_image()
    center_px, color = detect_block(img)
    if center_px is None:
        return None  # Nenhum bloco detectado
    
    x, y, z = pixel_to_world_ortho(*center_px)
    
    yaw = estimate_block_orientation(img, color)
    
    return x, y, z, yaw, color
    
# ==============================
# Função para exibir imagem e máscara continuamente
# ==============================
def display_camera_and_mask():
    """Captura e exibe a imagem da câmera e a máscara HSV com centro, coordenadas e yaw."""
    img = get_camera_image()
    if img is None:
        print("Erro: Não foi possível capturar a imagem da câmera.")
        return False, None, None, None, None, None
    
    img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))

    # Converter para HSV e detectar cor
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center_px, color = detect_block(img)

    # Exibir imagem original
    img_display = img.copy()
    x_world, y_world, z_world = None, None, None
    yaw = None

    if center_px and color:
        cx, cy = center_px
        # Desenhar centro na imagem original
        #cv2.circle(img_display, (cx, cy), 5, (0, 0, 0), -1)
        cv2.putText(img_display, f"Color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Calcular posição no mundo e yaw
        x_world, y_world, z_world = pixel_to_world_ortho(cx, cy)
        yaw = estimate_block_orientation(img, color)

        # Criar máscara e convertê-la para BGR
        color_ranges = {
            "red":   ([0, 200, 200], [10, 255, 255]),
            "green": ([50, 200, 200], [70, 255, 255]),
            "blue":  ([110, 200, 200], [130, 255, 255])
        }
        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Converter para BGR

        # Desenhar centro na máscara
        cv2.circle(mask_bgr, (cx, cy), 5, (0, 0, 255), -1)
        # Adicionar coordenadas e yaw
        text = f"X: {x_world:.3f}, Y: {y_world:.3f}, Z: {z_world:.3f}, Yaw: {math.degrees(yaw):.1f}deg"
        cv2.putText(mask_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Mask", mask_bgr)
        print(f"Máscara da cor {color} com centro e coords em ({x_world:.3f}, {y_world:.3f})")
    else:
        mask = np.zeros_like(img[:, :, 0])
        mask = cv2.resize(mask, (512, 512))  # Redimensionar a máscara vazia
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, "Nenhum bloco detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Mask", mask_bgr)
        print("Nenhuma cor detectada - mostrando máscara vazia.")

    cv2.imshow("Camera View", img_display)

    # Manter a janela responsiva
    key = cv2.waitKey(100)
    if key == 27:  # Pressionar 'Esc' para interromper
        return True, center_px, color, x_world, y_world, yaw
    return False, center_px, color, x_world, y_world, yaw
    
# ==============================
# Pipeline principal
# ==============================


if __name__ == "__main__":
    start_sim()
    set_gripper(True, 1.0)

    # --- CONFIGURAÇÃO ---
    pose_ready = (0.9, 0.0, 0.7, 180, 0, 90)
    deposit_locations = {
        "red":   (-0.8, 0.5, 0.08),
        "green": (-0.8, 0.0, 0.08),
        "blue":  (-0.8, -0.5, 0.08)
    }
    
    # --- Loop Principal ---
    while(True):
        print(f"\n\n--- INICIANDO NOVO CICLO ---")
        move_to_pose(*pose_ready)

        # 1. Spawnar
        print("\n--- 1. SPAWN DE BLOCO ---")
        bloco_atual = spawn_block()
        time.sleep(1)
        stop_signal, center_px, color, x, y, yaw_rad = display_camera_and_mask()

        # 2. Detectar
        print("\n--- 2. DETECÇÃO COM VISÃO ---")
        result = detect_block_world()
        if result is None or any(v is None for v in result):
             print("!! ERRO: Detecção falhou. Reiniciando ciclo.")
             despawn_block(bloco_atual)
             continue

        x, y, z, yaw_rad, color = result
        yaw_deg = math.degrees(yaw_rad)
        print(f"Bloco '{color}' detectado em (X={x:.3f}, Y={y:.3f}), Yaw={yaw_deg:.1f}°")
        
        # 3. Executar PICK
        print("\n--- 3. EXECUTANDO PICK ---")
        altura_segura_pick = 0.4
        altura_pegar = 0.025
        pose_acima_bloco = (x, y, altura_segura_pick, 180, 0, yaw_deg)
        pose_do_bloco = (x, y, altura_pegar, 180, 0, yaw_deg)
        
        move_to_pose(*pose_acima_bloco)
        move_to_pose(*pose_do_bloco)
        set_gripper(False)
        move_to_pose(*pose_acima_bloco)
        
        # --- 4. EXECUTANDO PLACE ---
        print("\n--- 4. EXECUTANDO PLACE ---")
        
        target_pos = deposit_locations[color]
        altura_segura_place = 0.35
        pose_acima_deposito = (target_pos[0], target_pos[1], altura_segura_place, 180, 0, 90)
        pose_do_deposito = (target_pos[0], target_pos[1], target_pos[2], 180, 0, 90)

        print(f"Transportando bloco '{color}'...")
        move_to_pose(*pose_ready)
        
        # --- LÓGICA DE TRAJETÓRIA LATERAL (COM 2 WAYPOINTS) ---
        waypoints_direita = [
            (0.6, 0.55, 0.75, 180, 0, 90),
            (0.0, 0.55, 0.75, 180, 0, 90)
        ]
        waypoints_esquerda = [
            (0.6, -0.55, 0.75, 180, 0, 90),
            (0.0, -0.55, 0.75, 180, 0, 90)
        ]
        
        caminho_de_ida = []
        if color == 'red':
            print("Usando trajetória lateral direita...")
            caminho_de_ida = waypoints_direita
        elif color == 'blue':
            print("Usando trajetória lateral esquerda...")
            caminho_de_ida = waypoints_esquerda
        elif color == 'green':
            if random.randint(0, 1) == 0:
                print("Sorteio: Usando trajetória lateral DIREITA para o bloco verde...")
                caminho_de_ida = waypoints_direita
            else:
                print("Sorteio: Usando trajetória lateral ESQUERDA para o bloco verde...")
                caminho_de_ida = waypoints_esquerda
        
        for wp in caminho_de_ida:
            move_to_pose(*wp)

        # Movimentos finais de depósito
        move_to_pose(*pose_acima_deposito)
        move_to_pose(*pose_do_deposito)
        set_gripper(True)
        time.sleep(0.5)
        despawn_block(bloco_atual)
        move_to_pose(*pose_acima_deposito)
        
        # --- 5. RETORNO PARA A POSIÇÃO READY ---
        print("\n--- 5. RETORNANDO À POSIÇÃO READY ---")
        
        # NOVO: Percorre o caminho de ida na ordem inversa
        caminho_de_volta = reversed(caminho_de_ida)
        for wp in caminho_de_volta:
            move_to_pose(*wp)

        print("\n--- CICLO COMPLETO ---")
        move_to_pose(*pose_ready)
        #cv2.destroyAllWindows()


"""
if __name__ == "__main__":
    start_sim()
    set_gripper(True, 1.0)

    # --- CONFIGURAÇÃO ---
    pose_ready = (0.9, 0.0, 0.7, 180, 0, 90)
    deposit_locations = {
        "red":   (-0.8, 0.5, 0.08),
        "green": (-0.8, 0.0, 0.08),
        "blue":  (-0.8, -0.5, 0.08)
    }
    
    # --- Loop Principal ---
    try: # Usamos um bloco try...finally para garantir que as janelas fechem
        while(True):
            print(f"\n\n--- INICIANDO NOVO CICLO ---")
            move_to_pose(*pose_ready)

            # 1. Spawnar
            print("\n--- 1. SPAWN DE BLOCO ---")
            bloco_atual = spawn_block()
            
            time.sleep(1)

            # 2. Detectar com visualização
            print("\n--- 2. DETECÇÃO COM VISÃO ---")
            # <<< ALTERAÇÃO AQUI >>> 
            # Trocamos detect_block_world() pela função com display
            stop_signal, center_px, color, x, y, yaw_rad = display_camera_and_mask()

            # <<< NOVO BLOCO DE VERIFICAÇÃO >>>
            # Verifica se o usuário pressionou 'Esc' para sair
            if stop_signal:
                print("Execução interrompida pelo usuário (tecla Esc).")
                despawn_block(bloco_atual) # Limpa o último bloco
                break # Sai do loop principal

            # Verifica se a detecção falhou (nenhum bloco encontrado)
            if color is None or x is None or yaw_rad is None:
                print("!! ERRO: Detecção falhou. Reiniciando ciclo.")
                despawn_block(bloco_atual)
                cv2.waitKey(1000) # Pausa para ver a mensagem na janela
                continue

            yaw_deg = math.degrees(yaw_rad)
            print(f"Bloco '{color}' detectado em (X={x:.3f}, Y={y:.3f}), Yaw={yaw_deg:.1f}°")
            
            # Pausa para o usuário ver o resultado da detecção antes do robô se mover
            #print("Pressione qualquer tecla na janela da câmera para continuar...")
            #cv2.waitKey(0) 

            # 3. Executar PICK
            print("\n--- 3. EXECUTANDO PICK ---")
            altura_segura_pick = 0.4
            altura_pegar = 0.025
            pose_acima_bloco = (x, y, altura_segura_pick, 180, 0, yaw_deg)
            pose_do_bloco = (x, y, altura_pegar, 180, 0, yaw_deg)
            
            move_to_pose(*pose_acima_bloco)
            move_to_pose(*pose_do_bloco)
            set_gripper(False)
            move_to_pose(*pose_acima_bloco)
            
            # --- 4. EXECUTANDO PLACE ---
            print("\n--- 4. EXECUTANDO PLACE ---")
            
            target_pos = deposit_locations[color]
            altura_segura_place = 0.35
            pose_acima_deposito = (target_pos[0], target_pos[1], altura_segura_place, 180, 0, 90)
            pose_do_deposito = (target_pos[0], target_pos[1], target_pos[2], 180, 0, 90)

            print(f"Transportando bloco '{color}'...")
            move_to_pose(*pose_ready)
            
            # --- LÓGICA DE TRAJETÓRIA LATERAL (COM 2 WAYPOINTS) ---
            waypoints_direita = [
                (0.6, 0.55, 0.75, 180, 0, 90),
                (0.0, 0.55, 0.75, 180, 0, 90)
            ]
            waypoints_esquerda = [
                (0.6, -0.55, 0.75, 180, 0, 90),
                (0.0, -0.55, 0.75, 180, 0, 90)
            ]
            
            caminho_de_ida = []
            if color == 'red':
                print("Usando trajetória lateral direita...")
                caminho_de_ida = waypoints_direita
            elif color == 'blue':
                print("Usando trajetória lateral esquerda...")
                caminho_de_ida = waypoints_esquerda
            elif color == 'green':
                if random.randint(0, 1) == 0:
                    print("Sorteio: Usando trajetória lateral DIREITA para o bloco verde...")
                    caminho_de_ida = waypoints_direita
                else:
                    print("Sorteio: Usando trajetória lateral ESQUERDA para o bloco verde...")
                    caminho_de_ida = waypoints_esquerda
            
            for wp in caminho_de_ida:
                move_to_pose(*wp)

            # Movimentos finais de depósito
            move_to_pose(*pose_acima_deposito)
            move_to_pose(*pose_do_deposito)
            set_gripper(True)
            time.sleep(0.5)
            despawn_block(bloco_atual)
            move_to_pose(*pose_acima_deposito)
            
            # --- 5. RETORNO PARA A POSIÇÃO READY ---
            print("\n--- 5. RETORNANDO À POSIÇÃO READY ---")
            
            caminho_de_volta = reversed(caminho_de_ida)
            for wp in caminho_de_volta:
                move_to_pose(*wp)

            print("\n--- CICLO COMPLETO ---")
            move_to_pose(*pose_ready)

    finally:
        # <<< ESSENCIAL >>> Garante que as janelas do OpenCV fechem ao final
        print("Finalizando... Fechando janelas e parando a simulação.")
        cv2.destroyAllWindows()
        time.sleep(1) # Pequena pausa para garantir que tudo fechou
        stop_sim()
"""
