import os
import sys

# TRUQUE DE MESTRE: Engana o sistema para ele não procurar a libGL física
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Configuração da IA
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("Biomecânica Caraguá 🎾")
st.write("Análise de vídeo por upload.")

video_file = st.file_uploader("Suba o vídeo aqui", type=['mp4', 'mov'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        st_frame.image(frame, channels="BGR")
    
    cap.release()
