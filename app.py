import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
# Adicione esta linha abaixo para garantir que as soluções sejam carregadas
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

# Agora, remova ou comente as linhas antigas que davam erro:
# mp_pose = mp.solutions.pose  <-- Pode apagar essa
# pose = mp_pose.Pose()        <-- Mantenha esta, mas veja o ajuste abaixo

# Ajuste a criação do objeto pose:
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Configurações iniciais da IA
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Função matemática (igual à anterior)
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0: angulo = 360 - angulo
    return angulo

# Classe que processa o vídeo do celular
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24") # Converte o frame do celular para o Python
        
        # Processamento da IA
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = pose.process(img_rgb)
        
        if resultado.pose_landmarks:
            landmarks = resultado.pose_landmarks.landmark
            
            # Pontos do Joelho Esquerdo (24, 26, 28)
            q = [landmarks[24].x, landmarks[24].y]
            j = [landmarks[26].x, landmarks[26].y]
            t = [landmarks[28].x, landmarks[28].y]
            
            angulo = calcular_angulo(q, j, t)
            
            # Lógica de Cor: Verde se agachou (<90°), Vermelho se está em pé
            cor = (0, 255, 0) if angulo < 90 else (0, 0, 255)
            
            # Desenha o esqueleto
            mp_drawing.draw_landmarks(img, resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=cor, thickness=3))
            
            # Escreve o ângulo na tela
            cv2.putText(img, f"Angulo: {int(angulo)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

# Interface do App no Streamlit
st.title("Analisador Biomecânico de Bolso 📱")
st.write("Aponte a câmera para o movimento lateral do aluno.")

webrtc_streamer(key="biomecanica", video_transformer_factory=VideoProcessor)
