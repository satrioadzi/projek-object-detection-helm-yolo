from ultralytics import YOLO
import cv2
import math
import os

# --- KONFIGURASI ---
model_path = 'best.pt'
source_video = "3691658-hd_1920_1080_30fps.mp4" 
classNames = ['helmet', 'no_helmet']

# --- PERUBAHAN 1: BUKA GERBANG LEBAR-LEBAR ---
# Turunkan confidence dasar serendah mungkin agar yang jauh terdeteksi.
# Coba 0.15 (15%) atau 0.20 (20%)
confidence_level = 0.15  

# --- PERUBAHAN 2: PAKAI "KACA PEMBESAR" (Resolusi Inference) ---
# Default YOLO adalah 640. Kita naikkan agar objek kecil terlihat detail.
# Pilihan: 640 (Cepat), 960 (Sedang), 1280 (Detail tapi berat)
inference_size = 960

# --- KONFIGURASI ANTI SALAH TUDUH ---
# Karena gerbang dibuka lebar (banyak sampah masuk), filter ini HARUS KUAT.
# Hanya cap "PELANGGAR" jika yakin di atas 70%. Sisanya dianggap AMAN.
strict_violator_conf = 0.70

# --- LOAD DATA ---
print(f"[INFO] Memuat model dan video (Inference Size: {inference_size})...")
if source_video != 0 and not os.path.exists(source_video):
    print(f"[ERROR] File '{source_video}' tidak ditemukan!")
    exit()

model = YOLO(model_path)
cap = cv2.VideoCapture(source_video)

# --- LOOPING ---
while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h_img, w_img, _ = img.shape

    # --- TRACKING DENGAN RESOLUSI TINGGI ---
    # Tambahkan parameter imgsz=inference_size
    results = model.track(img, persist=True, conf=confidence_level, imgsz=inference_size, verbose=False)

    jumlah_pelanggar = 0
    jumlah_patuh = 0

    # Ambil hasil
    r = results[0]
    boxes = r.boxes
    
    for box in boxes:
        # Koordinat
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Data
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls] if cls < len(classNames) else "Unknown"

        # ID Tracking
        if box.id is not None:
            track_id = int(box.id[0])
            id_text = f"#{track_id} "
        else:
            id_text = "" 

        # --- LOGIKA "ANTI SALAH TUDUH" (KETAT) ---
        is_violator = False
        
        # Cek potensi pelanggar
        if currentClass in ['no_helmet', 'head']:
            # Syaratnya LEBIH BERAT: Harus yakin > 70%
            if conf > strict_violator_conf:
                is_violator = True
            else:
                # Jika ragu (misal cuma 50%), paksa jadi AMAN
                is_violator = False
                currentClass = 'helmet' 
        
        # --- VISUALISASI WARNA ---
        if is_violator:
            color = (0, 0, 255) # Merah
            jumlah_pelanggar += 1
            label_text = f"{id_text}PELANGGAR {int(conf*100)}%"
        else:
            color = (0, 255, 0) # Hijau
            jumlah_patuh += 1
            label_text = f"{id_text}AMAN {int(conf*100)}%"

        # Gambar Kotak
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label Halus
        text_y = max(y1 - 5, 10)
        font_scale = 0.35
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(img, label_text, (x1, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label_text, (x1, text_y), font, font_scale, color, 1, cv2.LINE_AA)

    # --- STATUS BAWAH ---
    y_start = h_img - 30 
    font_status = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, f"PELANGGAR: {jumlah_pelanggar}", (21, y_start+1), font_status, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, f"PATUH: {jumlah_patuh}", (21, y_start - 29), font_status, 0.6, (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(img, f"PELANGGAR: {jumlah_pelanggar}", (20, y_start), font_status, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"PATUH: {jumlah_patuh}", (20, y_start - 30), font_status, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan
    cv2.imshow("Sistem Deteksi Helm (High Res Mode)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()