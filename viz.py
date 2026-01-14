import os
import glob
import numpy as np
import cv2
import open3d as o3d

# ===== USER CONFIG =====
LEFT_DIR = "/home/tong/recordings/uwstereo_test/tower/left2"
DISP_DIR = "/home/tong/recordings/uwstereo_test/tower/pred2/disparity"
EXT = "png"
MAX_DEPTH = 10.0  # set to e.g. 50.0 to clamp far points 
MIN_DEPTH = 0.5

scale_x = 800 / 1600
scale_y = 600 / 1200
# Camera intrinsics (pixels) + baseline (meters)
fx = 980.21 * scale_x
fy = 980.21 * scale_y
cx = 825.18 * scale_x
cy = 627.85 * scale_y
baseline = 0.1
# =======================

paused = False
should_exit = False


def depth_from_disp(disp, fx_value):
    depth = (fx_value * baseline) / disp 
    return depth


def resize_and_scale_intrinsics(img, target_hw, fx_value, fy_value, cx_value, cy_value):
    h0, w0 = img.shape[:2]
    h, w = target_hw
    sx = w / float(w0)
    sy = h / float(h0)
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized, fx_value * sx, fy_value * sy, cx_value * sx, cy_value * sy


def build_pointcloud(depth, color, fx_value, fy_value, cx_value, cy_value):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx_value) * z / fx_value
    y = -(v - cy_value) * z / fy_value

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def toggle_pause(vis):
    global paused
    paused = not paused
    return False


def request_exit(vis):
    global should_exit
    should_exit = True
    return False


left_files = sorted(glob.glob(os.path.join(LEFT_DIR, f"*.{EXT}")))
if not left_files:
    raise RuntimeError("No left images found.")

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud", width=960, height=720)
vis.register_key_callback(ord(' '), toggle_pause)
vis.register_key_callback(ord('q'), request_exit)
vis.register_key_callback(ord('Q'), request_exit)
vis.register_key_callback(256, request_exit)  # ESC

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

for lf in left_files:
    if should_exit:
        break

    if paused:
        vis.poll_events()
        vis.update_renderer()
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break
        continue

    stem = os.path.splitext(os.path.basename(lf))[0]
    disp_path = os.path.join(DISP_DIR, f"{stem}.npy")

    if not os.path.exists(disp_path):
        continue

    img_bgr = cv2.imread(lf)
    img = img_bgr[:, :, ::-1]  # RGB
    disp = np.load(disp_path)

    h, w = disp.shape
    img_small, fx_s, fy_s, cx_s, cy_s = resize_and_scale_intrinsics(img, (h, w), fx, fy, cx, cy)
    depth = depth_from_disp(disp, fx_s)
    depth[(depth <= 0) | (~np.isfinite(depth))] = np.nan

    if MAX_DEPTH is not None:
        depth[depth > MAX_DEPTH] = np.nan
    if MIN_DEPTH is not None:
        depth[depth < MIN_DEPTH] = np.nan

    # depth_vis = depth.copy()
    # depth_vis[~np.isfinite(depth_vis)] = 0
    # depth_vis = np.clip(depth_vis, 0, np.percentile(depth_vis, 98))
    # depth_vis = (depth_vis / (depth_vis.max() + 1e-6) * 255).astype(np.uint8)
    # depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # --- VISUALIZATION FIX ---
    depth_vis = depth.copy()
    
    # 1. Fill invalid pixels with the max depth (so they become blue background)
    max_val = np.nanpercentile(depth_vis, 98) 
    depth_vis[~np.isfinite(depth_vis)] = max_val
    depth_vis = np.clip(depth_vis, 0, max_val)

    # 2. Normalize to 0-255
    depth_vis = (depth_vis / (max_val + 1e-6) * 255).astype(np.uint8)

    # 3. *** INVERT HERE *** so Near (0) becomes Red (255)
    depth_vis = 255 - depth_vis 

    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # -------------------------

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.imshow("Left", cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR))
    cv2.resizeWindow("Left", w, h)
    cv2.imshow("Depth", depth_color)
    cv2.resizeWindow("Depth", w, h)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        paused = True

    pcd_new = build_pointcloud(depth, img_small, fx_s, fy_s, cx_s, cy_s)
    pcd.points = pcd_new.points
    pcd.colors = pcd_new.colors
    vis.update_geometry(pcd)
    if first_frame:
        vis.reset_view_point(True)
        first_frame = False
    vis.poll_events()
    vis.update_renderer()


# Keep windows open after last frame
while not should_exit:
    vis.poll_events()
    vis.update_renderer()
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break

