import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Inicialização da IA de forma segura
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("Analisador Biomecânico (Vídeo) 🎾")
st.write("Suba o vídeo do seu aluno para analisar o ângulo do joelho.")

video_file = st.file_uploader("Selecione o vídeo", type=['mp4', 'mov', 'avi'])

if video_file:
    # Cria um arquivo temporário para o OpenCV processar
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    video = cv2.VideoCapture(tfile.name)
    caixa_video = st.empty() # Local onde o vídeo processado vai aparecer

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Converte para RGB para o MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Desenha o esqueleto básico
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Feedback Visual
            cv2.putText(frame, "Processando Biomecanica...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostra o quadro atual no Streamlit
        caixa_video.image(frame, channels="BGR")
    
    video.release()
    st.success("Análise Finalizada!")
