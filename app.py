import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

class BiomecanicaProcessor(VideoProcessorBase):
    def __init__(self):
        # Carrega as ferramentas apenas quando o app inicia de fato
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Efeito espelho
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Feedback simples para teste
            cv2.putText(img, "IA ATIVA", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

st.title("Biomecânica Real-Time 🎾")
st.write("Aguarde carregar e clique em START.")

webrtc_streamer(
    key="teste-definitivo", 
    video_processor_factory=BiomecanicaProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
