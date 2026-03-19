import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np

# Configuração simples do MediaPipe (Padrão 2026)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0: angulo = 360 - angulo
    return angulo

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = pose.process(img_rgb)
        
        if resultado.pose_landmarks:
            landmarks = resultado.pose_landmarks.landmark
            
            # Quadril(24), Joelho(26), Tornozelo(28)
            q = [landmarks[24].x, landmarks[24].y]
            j = [landmarks[26].x, landmarks[26].y]
            t = [landmarks[28].x, landmarks[28].y]
            
            angulo = calcular_angulo(q, j, t)
            cor = (0, 255, 0) if angulo < 90 else (0, 0, 255)
            
            mp_drawing.draw_landmarks(img, resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=cor, thickness=3))
            
            cv2.putText(img, f"Angulo: {int(angulo)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

st.title("Analisador Biomecânico 🎾")
st.write("Dica: Use em local iluminado e de lado para a câmera.")

webrtc_streamer(key="biomecanica", video_transformer_factory=VideoProcessor)
