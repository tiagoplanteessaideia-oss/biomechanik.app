import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

class BiomecanicaProcessor(VideoProcessorBase):
    def __init__(self):
        # Carrega as ferramentas apenas quando o app inicia
        self.pose = mp.solutions.pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Espelho
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Cálculo do ângulo (simplificado para teste)
            landmarks = results.pose_landmarks.landmark
            # Quadril(24), Joelho(26), Tornozelo(28)
            y24, y26, y28 = landmarks[24].y, landmarks[26].y, landmarks[28].y
            cv2.putText(img, f"Joelho: {round(y26, 2)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

st.title("Biomecânica Real-Time 🎾")
webrtc_streamer(key="teste-final", video_processor_factory=BiomecanicaProcessor)
