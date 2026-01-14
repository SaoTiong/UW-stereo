import os
import cv2

# === CONFIG ===
AVI_PATH = "/home/tong/DWE/dwvo2avi/Tower_cam2.avi"
OUT_DIR = "/home/tong/recordings/uwstereo_test/tower/right"

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(AVI_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {AVI_PATH}")

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out_path = os.path.join(OUT_DIR, f"{idx:06d}.png")
    cv2.imwrite(out_path, frame)
    idx += 1

cap.release()
print(f"Saved {idx} frames to {OUT_DIR}")
