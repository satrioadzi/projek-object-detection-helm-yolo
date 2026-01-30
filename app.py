import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time
import numpy as np

# ======================================================
# CONFIG & SETUP
# ======================================================
st.set_page_config(page_title="Deteksi Helm AI", page_icon="🪖", layout="wide")

# Gunakan cache agar model tidak dimuat ulang setiap interaksi
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("⚙️ Kontrol Sistem")
mode = st.sidebar.selectbox("Pilih Mode", ["Deteksi Gambar", "Upload Video", "Webcam"])

st.sidebar.markdown("---")
st.sidebar.subheader("Parameter AI")

# Menjelaskan Strict Mode di UI agar Dosen lihat
conf_det = st.sidebar.slider("Min. Confidence Deteksi", 0.0, 1.0, 0.25, help="Ambang batas agar objek terdeteksi.")
conf_violate = st.sidebar.slider("Strict Mode Threshold", 0.0, 1.0, 0.70, help="Ambang batas keyakinan untuk memvonis PELANGGAR.")
imgsz = st.sidebar.select_slider("Resolusi Processing", [320, 480, 640], value=640)

# ======================================================
# CORE LOGIC (STRICT MODE)
# ======================================================
def draw_smart_box(img, box, cls, conf, class_names):
    x1, y1, x2, y2 = map(int, box)
    label = class_names[cls]
    
    # --- LOGIKA STRICT MODE (SESUAI MAKALAH) ---
    # Hanya vonis "PELANGGAR" jika confidence > conf_violate (misal 70%)
    # Jika terdeteksi 'no_helmet' tapi ragu-ragu (<70%), anggap AMAN (Benefit of the doubt)
    is_violation = (label == "no_helmet" and conf > conf_violate)

    if is_violation:
        color = (0, 0, 255) # Merah
        status_text = f"PELANGGAR {conf:.2f}"
    else:
        color = (0, 255, 0) # Hijau
        status_text = f"AMAN {conf:.2f}"

    # Visualisasi
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Background text agar terbaca jelas
    (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, status_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return is_violation

# ======================================================
# MAIN PROCESSING
# ======================================================
def process_frame(frame):
    # Resize untuk kecepatan inferensi (sesuai Bab 4 Makalah)
    frame_resized = cv2.resize(frame, (imgsz, int(frame.shape[0]*imgsz/frame.shape[1])))
    
    results = model.predict(frame_resized, conf=conf_det, verbose=False)[0]
    
    pelanggar_count = 0
    patuh_count = 0
    
    for box in results.boxes:
        # Ambil data bounding box
        coords = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Panggil fungsi visualisasi & hitung status
        is_violation = draw_smart_box(frame_resized, coords, cls, conf, results.names)
        
        if is_violation:
            pelanggar_count += 1
        else:
            patuh_count += 1
            
    return frame_resized, pelanggar_count, patuh_count

# ======================================================
# APP FLOW
# ======================================================
st.title("🪖 Sistem Monitoring Pelanggaran Helm")
st.markdown(f"**Status:** Menggunakan YOLOv8 dengan *Strict Mode Validation* (> {conf_violate*100:.0f}%)")

if mode == "Deteksi Gambar":
    file = st.file_uploader("Upload Gambar", ["jpg", "png", "jpeg"])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        processed_img, p, a = process_frame(img)
        
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.error(f"🔴 Terdeteksi Pelanggar: {p}")
        c2.success(f"🟢 Terdeteksi Patuh: {a}")

elif mode in ["Upload Video", "Webcam"]:
    src = 0 if mode == "Webcam" else None
    if mode == "Upload Video":
        file = st.file_uploader("Upload Video", ["mp4", "avi"])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            src = tfile.name

    if src is not None:
        if st.button("Mulai Deteksi"):
            cap = cv2.VideoCapture(src)
            st_frame = st.image([])
            kpi1, kpi2, kpi3 = st.columns(3)
            
            prev_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Proses Frame
                frame_out, p, a = process_frame(frame)
                
                # Hitung FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # Update UI
                kpi1.metric("🔴 Pelanggar", p)
                kpi2.metric("🟢 Patuh", a)
                kpi3.metric("⚡ FPS", f"{fps:.1f}")
                
                st_frame.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            cap.release()