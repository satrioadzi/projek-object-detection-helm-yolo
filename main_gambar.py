from ultralytics import YOLO
import cv2
import math
import os

# --- KONFIGURASI ---
model_path = 'best.pt'
nama_file_gambar = "test_image.jpg"  # Ganti dengan nama file Anda
strict_threshold = 0.70  # <--- INI KUNCI STRICT MODE (Sesuai Makalah)

# --- LOAD DATA ---
if not os.path.exists(nama_file_gambar):
    print(f"[ERROR] File '{nama_file_gambar}' tidak ditemukan!")
    exit()

img = cv2.imread(nama_file_gambar)
model = YOLO(model_path)

# --- PROSES DETEKSI ---
# Deteksi objek apa saja yang confidenya di atas 0.25 (Cukup sensitif)
results = model.predict(img, conf=0.25)[0] 
names = results.names

jumlah_pelanggar = 0
jumlah_patuh = 0

for box in results.boxes:
    # 1. Ambil Data
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    label_class = names[cls]

    # 2. LOGIKA STRICT MODE (CORE INNOVATION)
    # Default status: Aman
    is_violation = False
    
    if label_class == "no_helmet":
        # Cek apakah confidence cukup tinggi untuk memvonis pelanggaran?
        if conf > strict_threshold:
            is_violation = True
            jumlah_pelanggar += 1
            color = (0, 0, 255) # Merah
            label_text = f"PELANGGAR {int(conf*100)}%"
        else:
            # Jika 'no_helmet' tapi ragu (<70%), anggap AMAN (False Positive Handler)
            jumlah_patuh += 1
            color = (0, 255, 0) # Hijau
            label_text = f"AMAN (Low Conf) {int(conf*100)}%"
    else:
        # Jika terdeteksi 'helmet'
        jumlah_patuh += 1
        color = (0, 255, 0) # Hijau
        label_text = f"AMAN {int(conf*100)}%"

    # 3. VISUALISASI (Style Pro)
    # Kotak Bounding Box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label Text dengan Background outline (biar terbaca di background apapun)
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(label_text, font, font_scale, 1)
    
    # Background kotak untuk text
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    
    # Text Putih
    cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

# --- INFO PANEL ---
cv2.putText(img, f"STRICT MODE: > {int(strict_threshold*100)}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(img, f"TOTAL PELANGGAR: {jumlah_pelanggar}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# --- TAMPILKAN ---
cv2.imshow("Hasil Deteksi - Strict Mode", img)
cv2.waitKey(0)
cv2.destroyAllWindows()