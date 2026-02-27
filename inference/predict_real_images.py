import cv2
import numpy as np
from PIL import Image
import os
import glob
import natsort
import torch
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms
from feature_extractor import CNNFeatureExtractor
from transformer import SwathWidthTransformer

# -----------------------
# CAMERA CALIBRATION: 
fx_val=1058.27
fy_val=1057.97
cx_val=1148.22
cy_val=616.90
k1_val=-0.0398
k2_val=0.0078
p1_val=-0.0005
p2_val=-0.0003
k3_val=-0.0042
Xc=0.0
Yc=0.0
Zc=1.2
# Intrinsics (px) and distortion for our *validation* camera
K_val = np.array([[fx_val, 0, cx_val],
                  [0, fy_val, cy_val],
                  [0,      0,      1]], dtype=np.float32)
D_val = np.array([k1_val, k2_val, p1_val, p2_val, k3_val], dtype=np.float32)

# --- If you KNOW extrinsics (world Z-up; ground is Z=0): fill these; else leave and use markers path ---
# Camera position (meters) in world frame:
cam_pos_w = np.array([Xc, Yc, Zc], dtype=np.float32)  # e.g., [0.0, -12.0, 6.0]

# Camera orientation (yaw, pitch, roll in degrees). 
# Convention here: Z-up world; yaw (about Z), pitch (about X), roll (about Y), applied Rz * Rx * Ry
yaw_deg, pitch_deg, roll_deg = 0.0, -30.0, 0.0

# -----------------------
# ORTHO GRID (must match training)
# -----------------------
ORTHO_W_M = 60.0   # ground width (meters)
ORTHO_H_M = 60.0   # ground height (meters)
ORTHO_W_PX = 512
ORTHO_H_PX = 512
M_PER_PX = ORTHO_W_M / ORTHO_W_PX  # meters per pixel (X)
# Y uses the same (square pixels on ground)
assert abs(ORTHO_H_M / ORTHO_H_PX - M_PER_PX) < 1e-6, "Use square meter-pixels like during training."

# -----------------------
# HELPERS
# -----------------------
def euler_to_R(yaw_deg, pitch_deg, roll_deg):
    """Build world->camera rotation from yaw(Z), pitch(X), roll(Y) in degrees."""
    rz = np.deg2rad(yaw_deg); rx = np.deg2rad(pitch_deg); ry = np.deg2rad(roll_deg)
    Rz = np.array([[ np.cos(rz), -np.sin(rz), 0],
                   [ np.sin(rz),  np.cos(rz), 0],
                   [         0,           0, 1]], dtype=np.float32)
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(rx), -np.sin(rx)],
                   [0,  np.sin(rx),  np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [          0, 1,          0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    # World->cam rotation (choose a consistent convention; this one is common in CV)
    R = Rz @ Rx @ Ry  #matmultiplication
    return R

def undistort_frame(frame_bgr, K, D):
    """Undistort with OpenCV. Returns undistorted BGR and the effective K' used post-rectify. to remove the lens distortion"""
    h, w = frame_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)  # crop to valid
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
    undist = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)
    return undist, new_K

def img_to_grid_h_from_extrinsics(K_rect, R_wc, t_wc, grid_w_px, grid_h_px, m_per_px, grid_w_m, grid_h_m):
    """
    Build homography that maps *image pixels* -> *ortho grid pixels*.
    K_rect: intrinsics after undistort (3x3)
    R_wc, t_wc: world->camera rotation(3x3) and translation(3,), world Z-up; ground plane Z=0
    """
    # Camera extrinsics as camera matrix [R|t] mapping world->cam
    # For points on ground plane (Z=0): Xw = [X, Y, 0, 1]^T
    # Image projection: u ~ K * [r1 r2 t] * [X Y 1]^T
    H_g2i = K_rect @ np.c_[R_wc[:, 0], R_wc[:, 1], t_wc]  # ground->image
    H_i2g = np.linalg.inv(H_g2i)                           # image->ground (meters)

    # Ground (meters) -> grid (pixels). Origin at grid center, +X right, +Y up in world.
    # Grid u' = (X + Wm/2)/m_per_px ;  v' = (Hm/2 - Y)/m_per_px   (v' downward)
    S = 1.0 / m_per_px
    A_g2grid = np.array([[ S,  0,  grid_w_px * 0.5],
                         [ 0, -S,  grid_h_px * 0.5],
                         [ 0,  0,                 1]], dtype=np.float32)

    # Final: image -> grid pixels
    H_i2grid = A_g2grid @ H_i2g
    return H_i2grid

def img_to_grid_h_from_markers(src_pts_img_px, dst_pts_ground_m, grid_w_px, grid_h_px, m_per_px, grid_w_m, grid_h_m):
    """
    src_pts_img_px: Nx2 image pixels of known ground points
    dst_pts_ground_m: Nx2 ground metric coordinates (meters) in world frame, with Z=0.
                      Use world axes where +X right, +Y up. (You choose origin—e.g., grid center)
    Returns homography mapping image -> grid pixels.
    """
    src = np.asarray(src_pts_img_px, dtype=np.float32)
    # Convert ground meters -> grid pixels (same mapping as above)
    X = np.asarray(dst_pts_ground_m, dtype=np.float32)
    u = (X[:, 0] + grid_w_m * 0.5) / m_per_px
    v = (grid_h_m * 0.5 - X[:, 1]) / m_per_px
    dst = np.stack([u, v], axis=1).astype(np.float32)
    H_i2grid, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
    return H_i2grid

def warp_to_ortho_grid(frame_bgr, H_i2grid, out_w_px, out_h_px):
    """Warp image -> fixed ortho grid (pixels)."""
    top = cv2.warpPerspective(frame_bgr, H_i2grid, (out_w_px, out_h_px), flags=cv2.INTER_LINEAR)
    return top







# CONFIG
# -----------------------
frames_dir = "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames_process_6s"  # folder containing ~25 frames
weights_path = "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/best_model.pth"
backbone = 'resnet18'  # same backbone used in training
num_frames = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# MATCH TRAINING TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# -----------------------
# LOAD FEATURE EXTRACTOR & MODEL
# -----------------------
cnn_extractor = CNNFeatureExtractor(backbone=backbone, pretrained=True).to(device).eval()

model = SwathWidthTransformer(feature_dim=512, num_frames=num_frames).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# -----------------------
# EXTRACT FEATURES FROM 25 FRAMES
# -----------------------
frame_paths = natsort.natsorted(
    [p for p in glob.glob(os.path.join(frames_dir, "*"))
     if p.lower().endswith((".png", ".jpg", ".jpeg"))]
)

if len(frame_paths) == 0:
    raise RuntimeError(f"No frames found in {frames_dir}")

# pad or truncate to exactly 25 frames
if len(frame_paths) < num_frames:
    frame_paths += [frame_paths[-1]] * (num_frames - len(frame_paths))
else:
    frame_paths = frame_paths[:num_frames]

# -----------------------
# EXTRACT FEATURES FROM 25 FRAMES (orthorectified)
# -----------------------
# Choose ONE path below:
USE_EXTRINSICS = True   # if you have camera pose; else set False and fill markers

# If using markers (at least 4): fill these once (example)
# src_pts_img_px = [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]
# dst_pts_ground_m = [(X1,Y1), (X2,Y2), (X3,Y3), (X4,Y4)]  # meters on ground, Z=0

features_list = []

# Precompute homography once (faster)
H_i2grid_cached = None
K_rect_cached = None

with torch.no_grad():
    for idx, p in enumerate(frame_paths):
        # --- Load BGR with OpenCV, undistort, then warp to ortho ---
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")

        # 1) Undistort this camera's frame
        undist_bgr, K_rect = undistort_frame(bgr, K_val, D_val)
        #print(K_rect)
        #cv2.imwrite("/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/undistorded image.jpg",undist_bgr)
    

        # 2) C1ompute image->grid homography (once)
        if H_i2grid_cached is None or K_rect_cached is None:
            if USE_EXTRINSICS:
                # Build R and t (world->camera). 
                # t is camera center in camera coords: t = -R*cam_pos_w  (if world origin is on ground grid)
                R_wc = euler_to_R(yaw_deg, pitch_deg, roll_deg)
                t_wc = -R_wc @ cam_pos_w.reshape(3, 1)
                t_wc = t_wc.reshape(3)

                H_i2grid_cached = img_to_grid_h_from_extrinsics(
                    K_rect, R_wc, t_wc,
                    ORTHO_W_PX, ORTHO_H_PX, M_PER_PX, ORTHO_W_M, ORTHO_H_M
                )
         
            K_rect_cached = K_rect.copy()

        # 3) Warp to fixed ortho grid (BGR)
        ortho_bgr = warp_to_ortho_grid(bgr, H_i2grid_cached, ORTHO_W_PX, ORTHO_H_PX)
        ortho_bgr_path=os.path.join("/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames_process_6s", "ortho_img.png")
        cv2.imwrite(ortho_bgr_path,ortho_bgr)

        # 4) Convert to PIL RGB and apply your *training* transforms
        ortho_rgb = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(ortho_rgb)
        x = transform(img).unsqueeze(0).to(device)  # [1,3,512,512]

        # 5) Extract features (512)
        feat = cnn_extractor(x)  # [1,512]
        features_list.append(feat.squeeze(0))

features_25x512 = torch.stack(features_list, dim=0).unsqueeze(0).to(device)  # [1,25,512]

# -----------------------
# PREDICT (keep your scale if you truly need it)
# -----------------------
with torch.no_grad():
    pred_width_m = model(features_25x512).item()

print(f"🎯 Predicted swath width: {pred_width_m:.4f} m")



# -----------------------
# load original last frame at its native resolution so the overlay looks crisp
path= "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames_process_6s/frame_0098.png" 
img=cv2.imread(path)
resized_img=cv2.resize(img,(512,512))
cv2.imwrite("/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames_process_6s/frame_resized.png",resized_img)
path= "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames_process_6s/frame_resized.png" 
base_img = Image.open(path).convert("RGB")
draw = ImageDraw.Draw(base_img)

# choose a font (fallback to default if truetype not available)
try:
    # You can point to a specific .ttf on your system if you prefer
    font_size = max(22, base_img.width // 32)  # scale with image width
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
except Exception:
    font = ImageFont.load_default()
    font_size = 22 # approximate

text = f"Predicted swath width: {pred_width_m :.2f} m"

# compute text size and draw a semi-transparent rectangle behind for readability
text_bbox = draw.textbbox((0, 0), text, font=font)
text_w = text_bbox[2] - text_bbox[0]
text_h = text_bbox[3] - text_bbox[1]
pad = max(6, font_size // 3)

# bottom-left corner placement
x = pad
y = base_img.height - text_h - 2*pad

# rectangle background
bg_rect = [x - pad, y - pad, x + text_w + pad, y + text_h + pad]
# draw opaque (PIL RGB). To simulate transparency on RGB, use a darker box and light text.
draw.rectangle(bg_rect, fill=(0, 0, 0))
draw.text((x, y), text, fill=(255, 255, 255), font=font)

# save next to frames as a new file

out_path = os.path.join(os.path.dirname(path), "prediction.png")
base_img.save(out_path)
print(f"💾 Saved annotated last frame to: {out_path}")
