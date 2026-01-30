import streamlit as st

from ultralytics import YOLO

import cv2

import tempfile

import time

import numpy as np



# ======================================================

# CONFIG

# ======================================================

st.set_page_config(page_title="Deteksi Helm AI", page_icon="🪖", layout="wide")



CLASS_NAMES = ['helmet', 'no_helmet']



# ======================================================

# LOAD MODEL

# ======================================================

@st.cache_resource

def load_model():

    return YOLO("best.pt")



model = load_model()



# ======================================================

# UI

# ======================================================

st.title("🪖 Sistem Monitoring Pelanggaran Helm")



mode = st.sidebar.selectbox(

    "Mode",

    ["Deteksi Gambar", "Upload Video", "Webcam"]

)



conf_det = st.sidebar.slider("Confidence Deteksi", 0.1, 1.0, 0.35, 0.05)

conf_violate = st.sidebar.slider("Confidence Pelanggar", 0.1, 1.0, 0.45, 0.05)

imgsz = st.sidebar.select_slider("Resolusi", [320, 480, 640, 960], value=640)



# ======================================================

# DRAWING

# ======================================================

def draw_box(img, box, cls, conf):

    x1, y1, x2, y2 = map(int, box)

    label = CLASS_NAMES[cls]



    if label == "no_helmet" and conf > conf_violate:

        color = (0, 0, 255)

        text = f"PELANGGAR {conf:.2f}"

    else:

        color = (0, 255, 0)

        text = f"AMAN {conf:.2f}"



    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(img, text, (x1, y1 - 6),

                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# ======================================================

# IMAGE

# ======================================================

def process_image(file):

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)



    result = model.predict(

        img,

        conf=conf_det,

        imgsz=imgsz,

        device="cpu",

        verbose=False

    )[0]



    pelanggar, patuh = 0, 0



    for box, cls, conf in zip(

        result.boxes.xyxy,

        result.boxes.cls,

        result.boxes.conf

    ):

        draw_box(img, box, int(cls), float(conf))

        if CLASS_NAMES[int(cls)] == "no_helmet" and conf > conf_violate:

            pelanggar += 1

        else:

            patuh += 1



    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.error(f"🔴 Pelanggar: {pelanggar}")

    st.success(f"🟢 Patuh: {patuh}")



# ======================================================

# VIDEO / WEBCAM

# ======================================================

def process_video(src):

    cap = cv2.VideoCapture(src)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)



    st_frame = st.image([])

    col1, col2, col3 = st.columns(3)

    k1, k2, k3 = col1.empty(), col2.empty(), col3.empty()



    frame_id = 0

    prev_time = time.time()



    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break



        frame_id += 1



        # ⏩ SKIP FRAME (BIAR MULUS)

        if frame_id % 2 != 0:

            continue



        frame = cv2.resize(frame, (imgsz, int(frame.shape[0]*imgsz/frame.shape[1])))



        result = model.predict(

            frame,

            conf=conf_det,

            imgsz=imgsz,

            device="cpu",

            verbose=False

        )[0]



        pelanggar, patuh = 0, 0



        for box, cls, conf in zip(

            result.boxes.xyxy,

            result.boxes.cls,

            result.boxes.conf

        ):

            draw_box(frame, box, int(cls), float(conf))

            if CLASS_NAMES[int(cls)] == "no_helmet" and conf > conf_violate:

                pelanggar += 1

            else:

                patuh += 1



        fps = 1 / (time.time() - prev_time)

        prev_time = time.time()



        k1.metric("🔴 Pelanggar", pelanggar)

        k2.metric("🟢 Patuh", patuh)

        k3.metric("⚡ FPS", f"{fps:.1f}")



        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),

                       use_container_width=True)



    cap.release()



# ======================================================

# MAIN

# ======================================================

if mode == "Deteksi Gambar":

    file = st.file_uploader("Upload Gambar", ["jpg", "png"])

    if file:

        process_image(file)



elif mode == "Upload Video":

    file = st.file_uploader("Upload Video", ["mp4", "avi"])

    if file:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(file.read())

        process_video(tfile.name)



elif mode == "Webcam":

    process_video(0)