import os
import cv2
from tqdm import tqdm

# === CONFIG ===
AVI_PATH = "/home/tong/DWE/dwvo2avi/stereo_2025-11-14_13-52-51_PST_964000_part2_cam2.avi"
OUT_DIR = "/home/tong/recordings/uwstereo_test/miramar_lake/right_raw"

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(AVI_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {AVI_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = 0
with tqdm(total=total_frames, unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(OUT_DIR, f"{idx:06d}.png")
        cv2.imwrite(out_path, frame)
        idx += 1
        pbar.update(1)

cap.release()
print(f"Saved {idx} frames to {OUT_DIR}")
