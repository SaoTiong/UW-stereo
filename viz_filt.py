import os
import glob
import numpy as np
import cv2
import open3d as o3d

# ===== USER CONFIG =====
LEFT_DIR = "/home/tong/recordings/uwstereo_test/tower/left1"
DISP_DIR = "/home/tong/recordings/uwstereo_test/miramar_lake/pred10000/disparity"
COV_DIR = "/home/tong/recordings/uwstereo_test/miramar_lake/pred10000/covariance"
EXT = "png"
COV_THRESH = 1
MAX_DEPTH = 1
MIN_DEPTH = 0
DISPLAY_MAX_DEPTH = 3

TEXTURE_THRESHOLD = 1
MASK_CLOSE_KERNEL = (5, 5)
MASK_OPEN_KERNEL = (3, 3)

scale_x = 800 / 1600
scale_y = 600 / 1200
# Camera intrinsics (pixels) + baseline (meters)
fx = 980.21 * scale_x
fy = 980.21 * scale_y
cx = 825.18 * scale_x
cy = 627.85 * scale_y
baseline = 0.1
# =======================

TRANSFORM_MATRIX = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
])

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
    y = (v - cy_value) * z / fy_value

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def texture_filter(image_rgb, disparity_map, threshold):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    mask = gradient_magnitude > threshold

    kernel_close = np.ones(MASK_CLOSE_KERNEL, np.uint8)
    mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

    kernel_open = np.ones(MASK_OPEN_KERNEL, np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel_open)

    final_mask = mask_cleaned > 0
    filtered_disp = disparity_map.copy()
    filtered_disp[~final_mask] = 0
    return filtered_disp, final_mask


def pause(vis):
    global paused
    paused = True
    return False


def resume(vis):
    global paused
    paused = False
    return False


def request_exit(vis):
    global should_exit
    should_exit = True
    return False


left_files = sorted(glob.glob(os.path.join(LEFT_DIR, f"*.{EXT}")))
if not left_files:
    raise RuntimeError("No left images found.")

cv2.namedWindow("View", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Texture Thresh", "View", TEXTURE_THRESHOLD, 255, lambda x: None)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud", width=960, height=720)
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.0, 0.0, 0.0])
vis.register_key_callback(ord('q'), request_exit)
vis.register_key_callback(ord('Q'), request_exit)
vis.register_key_callback(256, request_exit)  # ESC
vis.register_key_callback(ord('s'), pause)
vis.register_key_callback(ord('r'), resume)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

for lf in left_files:
    if should_exit:
        break

    if paused:
        vis.poll_events()
        vis.update_renderer()
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            paused = False
        continue

    stem = os.path.splitext(os.path.basename(lf))[0]
    disp_path = os.path.join(DISP_DIR, f"{stem}.npy")
    cov_path = os.path.join(COV_DIR, f"{stem}.npy")

    if not os.path.exists(disp_path):
        continue
    if COV_DIR and not os.path.exists(cov_path):
        continue

    img_bgr = cv2.imread(lf)
    img = img_bgr[:, :, ::-1]  # RGB
    disp = -np.load(disp_path)
    cov = np.load(cov_path) if COV_DIR else None

    h, w = disp.shape
    img_small, fx_s, fy_s, cx_s, cy_s = resize_and_scale_intrinsics(img, (h, w), fx, fy, cx, cy)

    current_thresh = cv2.getTrackbarPos("Texture Thresh", "View")
    disp_filtered, mask_result = texture_filter(img_small, disp, current_thresh)
    depth = depth_from_disp(disp_filtered, fx_s)
    depth[(depth <= 0) | (~np.isfinite(depth))] = np.nan
    if cov is not None:
        depth[cov > COV_THRESH] = np.nan

    if MAX_DEPTH is not None:
        depth[depth > MAX_DEPTH] = np.nan
    if MIN_DEPTH is not None:
        depth[depth < MIN_DEPTH] = np.nan

    depth_vis = depth.copy()
    invalid_mask = ~np.isfinite(depth_vis)
    depth_vis[invalid_mask] = 0
    depth_vis = np.clip(depth_vis, 0, DISPLAY_MAX_DEPTH)
    depth_vis = (depth_vis / DISPLAY_MAX_DEPTH * 255).astype(np.uint8)
    depth_vis = 255 - depth_vis
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_color[invalid_mask] = (0, 0, 0)

    left_small = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    combined = cv2.hconcat([left_small, depth_color])
    cv2.namedWindow("View", cv2.WINDOW_NORMAL)
    cv2.imshow("View", combined)
    cv2.resizeWindow("View", combined.shape[1], combined.shape[0])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        paused = True
    if key == ord('r'):
        paused = False

    pcd_new = build_pointcloud(depth, img_small, fx_s, fy_s, cx_s, cy_s)
    pcd_new.transform(TRANSFORM_MATRIX)
    pcd.points = pcd_new.points
    pcd.colors = pcd_new.colors
    vis.update_geometry(pcd)
    if first_frame:
        vis.reset_view_point(True)
        first_frame = False
    vis.poll_events()
    vis.update_renderer()


while not should_exit:
    vis.poll_events()
    vis.update_renderer()
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break
