import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR = '/home/tong/datasets/UW-Stereo/output_40000'

IMG_LEFT_DIR = os.path.join(ROOT_DIR, 'image_left')
IMG_RIGHT_DIR = os.path.join(ROOT_DIR, 'image_right')
DISP_DIR = os.path.join(ROOT_DIR, 'depth_left')  # .npy disparity

print(f"Scanning for corrupted files in: {ROOT_DIR}...")

left_files = sorted(glob.glob(os.path.join(IMG_LEFT_DIR, '*.png')))
print(f"Found {len(left_files)} left images.")

corrupted_count = 0

for left_path in tqdm(left_files):
    filename = os.path.basename(left_path)
    file_id = os.path.splitext(filename)[0]

    right_path = os.path.join(IMG_RIGHT_DIR, filename)
    disp_path = os.path.join(DISP_DIR, file_id + '.npy')

    is_corrupt = False

    # 1) Left image readable?
    try:
        with Image.open(left_path) as img:
            img.verify()
    except Exception:
        is_corrupt = True
        print(f"\n[Bad File] Left image corrupted: {filename}")

    # 2) Right image exists + readable?
    if not is_corrupt:
        if not os.path.exists(right_path):
            is_corrupt = True
            print(f"\n[Bad File] Missing right image for: {filename}")
        else:
            try:
                with Image.open(right_path) as img:
                    img.verify()
            except Exception:
                is_corrupt = True
                print(f"\n[Bad File] Right image corrupted: {filename}")

    # 3) Disparity exists?
    if not is_corrupt and not os.path.exists(disp_path):
        is_corrupt = True
        print(f"\n[Bad File] Missing disparity for: {filename}")

    # 4) Disparity npy readable?
    if not is_corrupt:
        try:
            arr = np.load(disp_path)
            if arr.size == 0:
                is_corrupt = True
                print(f"\n[Bad File] Disparity npy empty: {filename}")
        except Exception:
            is_corrupt = True
            print(f"\n[Bad File] Disparity npy corrupted: {filename}")

    # === DELETE IF CORRUPT ===
    if is_corrupt:
        corrupted_count += 1
        try:
            if os.path.exists(left_path):
                os.remove(left_path)
            if os.path.exists(right_path):
                os.remove(right_path)
            if os.path.exists(disp_path):
                os.remove(disp_path)
            print(f"   -> DELETED group: {filename}")
        except OSError as e:
            print(f"   -> Error deleting {filename}: {e}")

print(f"\nScan complete. Deleted {corrupted_count} corrupted sets.")
