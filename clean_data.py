import os
import glob
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
# Double check this path matches your real folder
ROOT_DIR = '/home/tong/datasets/UW-Stereo/output_40000'

# Define your subfolders based on your previous errors
# (Adjust these if your folder names are different)
IMG_LEFT_DIR = os.path.join(ROOT_DIR, 'image_left')
IMG_RIGHT_DIR = os.path.join(ROOT_DIR, 'image_right')
DISP_DIR = os.path.join(ROOT_DIR, 'depth_left')  # You mentioned .npy is here

print(f"Scanning for corrupted images in: {IMG_LEFT_DIR}...")

# Get list of all left images
left_files = sorted(glob.glob(os.path.join(IMG_LEFT_DIR, '*.png')))
print(f"Found {len(left_files)} images total.")

corrupted_count = 0

for left_path in tqdm(left_files):
    filename = os.path.basename(left_path)
    file_id = filename.split('.')[0] # e.g., gets "32000" from "32000.png"
    
    # Construct paths for corresponding files
    right_path = os.path.join(IMG_RIGHT_DIR, filename)
    
    # Check for .npy first (as you requested), but also check .pfm/.png just in case
    disp_path = os.path.join(DISP_DIR, file_id + '.npy')
    if not os.path.exists(disp_path):
        disp_path = os.path.join(DISP_DIR, file_id + '.pfm')
    if not os.path.exists(disp_path):
        disp_path = os.path.join(DISP_DIR, file_id + '.png')

    is_corrupt = False

    # 1. Check if Left Image is readable
    try:
        with Image.open(left_path) as img:
            img.verify() # fast check
    except Exception:
        is_corrupt = True
        print(f"\n[Bad File] Left image corrupted: {filename}")

    # 2. Check if Right Image exists and is readable
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

    # 3. Check if Disparity exists
    if not is_corrupt and not os.path.exists(disp_path):
        is_corrupt = True
        print(f"\n[Bad File] Missing disparity for: {filename}")

    # === DELETE IF CORRUPT ===
    if is_corrupt:
        corrupted_count += 1
        try:
            if os.path.exists(left_path): os.remove(left_path)
            if os.path.exists(right_path): os.remove(right_path)
            if os.path.exists(disp_path): os.remove(disp_path)
            print(f"   -> DELETED group: {filename}")
        except OSError as e:
            print(f"   -> Error deleting {filename}: {e}")

print(f"\nScan complete. Deleted {corrupted_count} corrupted sets.")