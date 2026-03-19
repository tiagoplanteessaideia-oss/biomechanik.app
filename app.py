import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Configuração da IA
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0: angulo = 360 - angulo
    return angulo

st.title("Analisador Biomecânico de Vídeo 🎥")
st.info("Filme o movimento lateralmente e faça o upload abaixo.")

arquivo_video = st.file_uploader("Escolha um vídeo (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if arquivo_video is not None:
    # Salva o vídeo temporariamente para o OpenCV ler
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(arquivo_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # Espaço vazio para o vídeo aparecer

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processamento da Biomecânica
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Quadril(24), Joelho(26), Tornozelo(28)
            q = [landmarks[24].x, landmarks[24].y]
            j = [landmarks[26].x, landmarks[26].y]
            t = [landmarks[28].x, landmarks[28].y]
            
            angulo = calcular_angulo(q, j, t)
            
            # Desenha o esqueleto
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angulo: {int(angulo)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostra no Streamlit
        st_frame.image(frame, channels="BGR")

    cap.release()
    st.success("Análise concluída!")
