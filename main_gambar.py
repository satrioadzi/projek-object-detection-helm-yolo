from ultralytics import YOLO
import cv2
import math
import os

# --- KONFIGURASI ---
model_path = 'best.pt'
nama_file_gambar = "pexels-k3ithvision-9475868.jpg" # <--- Sesuaikan nama file
classNames = ['helmet', 'no_helmet']
confidence_level = 0.5

# --- 1. LOAD DATA ---
print(f"[INFO] Memuat gambar: {nama_file_gambar}...")
if not os.path.exists(nama_file_gambar):
    print(f"[ERROR] File '{nama_file_gambar}' tidak ditemukan!")
    exit()

img = cv2.imread(nama_file_gambar)
h_img, w_img, _ = img.shape
model = YOLO(model_path)

# --- 2. PROSES DETEKSI ---
results = model.predict(img, conf=confidence_level)

jumlah_pelanggar = 0
jumlah_patuh = 0

# --- 3. VISUALISASI DETEKSI ---
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Koordinat Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Data
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls] if cls < len(classNames) else "Unknown"

        # Logika Warna & Label
        if currentClass in ['no_helmet', 'head']:
            color = (0, 0, 255) # Merah
            jumlah_pelanggar += 1
            label_text = f"PELANGGAR {int(conf*100)}%"
        elif currentClass == 'helmet':
            color = (0, 255, 0) # Hijau
            jumlah_patuh += 1
            label_text = f"AMAN {int(conf*100)}%"
        else:
            color = (255, 0, 0)
            label_text = currentClass

        # 1. Gambar Kotak (Tipis)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 2. Label Transparan (High Quality)
        text_y = max(y1 - 5, 10)
        
        # --- PERBAIKAN FONT DISINI ---
        # Kita pakai scale 0.35 supaya hurufnya punya "ruang" untuk dibaca
        font_scale = 0.35 
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # LINE_AA membuat tulisan menjadi halus (tidak kotak-kotak/pecah)
        # Outline Hitam (Ketebalan 2)
        cv2.putText(img, label_text, (x1, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        # Warna Asli (Ketebalan 1) di atasnya
        cv2.putText(img, label_text, (x1, text_y), font, font_scale, color, 1, cv2.LINE_AA)

# --- 4. STATUS POJOK KIRI BAWAH ---
y_start = h_img - 30 
font_status = cv2.FONT_HERSHEY_SIMPLEX

# Shadow Hitam Status (Pakai LINE_AA juga biar rapi)
cv2.putText(img, f"PELANGGAR: {jumlah_pelanggar}", (21, y_start+1), font_status, 0.6, (0,0,0), 3, cv2.LINE_AA)
cv2.putText(img, f"PATUH: {jumlah_patuh}", (21, y_start - 29), font_status, 0.6, (0,0,0), 3, cv2.LINE_AA)

# Teks Status Asli
cv2.putText(img, f"PELANGGAR: {jumlah_pelanggar}", (20, y_start), font_status, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(img, f"PATUH: {jumlah_patuh}", (20, y_start - 30), font_status, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# Print laporan text di terminal
print(f"Status: {jumlah_pelanggar} Pelanggar, {jumlah_patuh} Patuh.")

# --- TAMPILKAN HASIL ---
nama_jendela = "Hasil Deteksi Helm"
cv2.namedWindow(nama_jendela, cv2.WINDOW_NORMAL) 
cv2.imshow(nama_jendela, img)

cv2.waitKey(0)
cv2.destroyAllWindows()