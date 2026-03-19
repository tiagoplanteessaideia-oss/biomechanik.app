import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

# Função para o cálculo biomecânico
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0: angulo = 360 - angulo
    return angulo

class BiomecanicaProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inverte para efeito espelho
        img = cv2.flip(img, 1)
        
        # Converte para RGB para o MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = self.pose.process(img_rgb)
        
        if resultado.pose_landmarks:
            landmarks = resultado.pose_landmarks.landmark
            
            # Pontos: Quadril(24), Joelho(26), Tornozelo(28)
            try:
                q = [landmarks[24].x, landmarks[24].y]
                j = [landmarks[26].x, landmarks[26].y]
                t = [landmarks[28].x, landmarks[28].y]
                
                angulo = calcular_angulo(q, j, t)
                cor = (0, 255, 0) if angulo < 100 else (0, 0, 255)
                
                # Desenha o esqueleto
                self.mp_drawing.draw_landmarks(
                    img, 
                    resultado.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=cor, thickness=3)
                )
                
                # Texto do ângulo
                cv2.putText(img, f"Angulo: {int(angulo)}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            except:
                pass

        return frame.from_ndarray(img, format="bgr24")

st.title("Analisador Biomecânico Pro 🎾")
st.write("Aguarde a câmera carregar e clique em 'Start'.")

webrtc_streamer(
    key="biomecanica-v1", 
    video_processor_factory=BiomecanicaProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
