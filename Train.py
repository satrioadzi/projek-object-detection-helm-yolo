from ultralytics import YOLO

# 1. Pilih model dasar (kita pakai Nano supaya ringan di laptop)
model = YOLO('yolov8n.pt') 

# 2. Mulai Training
# 'data.yaml' adalah file yang kamu dapat dari download Roboflow tadi
# epochs=20 artinya model akan belajar mengulang materi sebanyak 20 kali
results = model.train(data='data.yaml', epochs=20, imgsz=640)