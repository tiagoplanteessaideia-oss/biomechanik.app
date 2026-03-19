import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

# Função matemática de apoio
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0: angulo = 360 - angulo
    return angulo

# A Classe que gerencia o vídeo
class BiomecanicaProcessor(VideoProcessorBase):
    def __init__(self):
        # Iniciamos a IA aqui dentro, de forma protegida
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inverte a imagem para parecer um espelho (melhor para o aluno)
        img = cv2.flip(img, 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = self.pose.process(img_rgb)
        
        if resultado.pose_landmarks:
            landmarks = resultado.pose_landmarks.landmark
            
            # Coordenadas dos pontos 24, 26 e 28
            q = [landmarks[24].x, landmarks[24].y]
            j = [landmarks[26].x, landmarks[26].y]
            t = [landmarks[28].x, landmarks[28].y]
            
            angulo = calcular_angulo(q, j, t)
            
            # Feedback visual: Verde para agachado, Vermelho para em pé
            cor = (0, 255, 0) if angulo < 100 else (0, 0, 255)
            
            # Desenha as conexões
            self.mp_drawing.draw_landmarks(
                img, 
                resultado.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=cor, thickness=3)
            )
            
            # Escreve o ângulo na tela
            cv2.putText(img, f"Angulo: {int(angulo)} deg", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# Interface do Streamlit
st.set_page_config(page_title="Analisador Biomecânico", layout="wide")
st.title("Analisador Biomecânico 🎾")
st.write("Aponte a câmera lateralmente para o aluno.")

# Chamada do WebRTC (o coração do app no celular)
webrtc_streamer(
    key="biomecanica", 
    video_processor_factory=BiomecanicaProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Ajuda a conectar no 4G/5G
    media_stream_constraints={"video": True, "audio": False}
)
