import os
import glob
import natsort
import torch
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms
from feature_extractor import CNNFeatureExtractor
from transformer import SwathWidthTransformer
import cv2

# -----------------------
# CONFIG
# -----------------------
frames_dir = "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames"  # folder containing ~25 frames
weights_path = "/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/best_model.pth"
backbone = 'resnet18'  # same backbone used in training
num_frames = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F_train=60
F_new= 2.90

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

features_list = []
with torch.no_grad():
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)  # [1, 3, 512, 512]
        feat = cnn_extractor(x)                     # [1, 512]
        features_list.append(feat.squeeze(0))

features_25x512 = torch.stack(features_list, dim=0).unsqueeze(0).to(device)  # [1, 25, 512]

# -----------------------
# PREDICT
# -----------------------
with torch.no_grad():
    pred_width_m = model(features_25x512).item()

print(f"🎯 Predicted swath width: {pred_width_m:.4f} m")


# DRAW PREDICTION ON LAST FRAME & SAVE
# -----------------------
# load original last frame at its native resolution so the overlay looks crisp
last_frame_path="/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/frames/frame_0003.png"


# DRAW PREDICTION ON LAST FRAME & SAVE
# -----------------------
# load original last frame at its native resolution so the overlay looks crisp
base_img = Image.open(last_frame_path).convert("RGB")

draw = ImageDraw.Draw(base_img)

# choose a font (fallback to default if truetype not available)
try:
    # You can point to a specific .ttf on your system if you prefer
    font_size = max(22, base_img.width // 32)  # scale with image width
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
except Exception:
    font = ImageFont.load_default()
    font_size = 22 # approximate

text = f"Predicted swath width: {pred_width_m:.2f} m"

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
out_path = os.path.join(os.path.dirname(last_frame_path), "last_with_pred.png")
base_img.save(out_path)
print(f"💾 Saved annotated last frame to: {out_path}")