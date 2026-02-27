import argparse
import os
import glob
import natsort
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.feature_extractor import CNNFeatureExtractor
from models.transformer_encoder import SwathWidthTransformer


# =======================================================
# ARGUMENT PARSER
# =======================================================
parser = argparse.ArgumentParser(description="Predict swath width from synthetic frames")

parser.add_argument('--frames_dir', type=str, required=True,
                    help='Directory containing sequential frames')
parser.add_argument('--weights', type=str, required=True,
                    help='Path to trained model weights (.pth)')
parser.add_argument('--output_dir', type=str, default='inference_results',
                    help='Directory to save annotated image')
parser.add_argument('--num_frames', type=int, default=25)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================================================
# TRANSFORM (same as training)
# =======================================================
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


# =======================================================
# LOAD MODEL
# =======================================================
cnn_extractor = CNNFeatureExtractor(backbone='resnet18', pretrained=True).to(device).eval()

model = SwathWidthTransformer(feature_dim=512, num_frames=args.num_frames).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()


# =======================================================
# LOAD FRAMES
# =======================================================
frame_paths = natsort.natsorted(
    [p for p in glob.glob(os.path.join(args.frames_dir, "*"))
     if p.lower().endswith((".png", ".jpg", ".jpeg"))]
)

if len(frame_paths) == 0:
    raise RuntimeError(f"No frames found in {args.frames_dir}")

# pad or truncate
if len(frame_paths) < args.num_frames:
    frame_paths += [frame_paths[-1]] * (args.num_frames - len(frame_paths))
else:
    frame_paths = frame_paths[:args.num_frames]


# =======================================================
# FEATURE EXTRACTION
# =======================================================
features_list = []

with torch.no_grad():
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        feat = cnn_extractor(x)
        features_list.append(feat.squeeze(0))

features = torch.stack(features_list).unsqueeze(0).to(device)


# =======================================================
# PREDICTION
# =======================================================
with torch.no_grad():
    pred_width = model(features).item()

print(f"🎯 Predicted swath width: {pred_width:.3f} m")


# =======================================================
# SAVE ANNOTATED LAST FRAME
# =======================================================
last_frame_path = frame_paths[-1]
base_img = Image.open(last_frame_path).convert("RGB")
draw = ImageDraw.Draw(base_img)

try:
    font_size = max(22, base_img.width // 30)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
except:
    font = ImageFont.load_default()
    font_size = 22

text = f"Predicted swath width: {pred_width:.2f} m"

text_bbox = draw.textbbox((0, 0), text, font=font)
text_w = text_bbox[2] - text_bbox[0]
text_h = text_bbox[3] - text_bbox[1]
pad = max(6, font_size // 3)

x = pad
y = base_img.height - text_h - 2*pad

bg_rect = [x - pad, y - pad, x + text_w + pad, y + text_h + pad]
draw.rectangle(bg_rect, fill=(0, 0, 0))
draw.text((x, y), text, fill=(255, 255, 255), font=font)

output_path = os.path.join(args.output_dir, "synthetic_prediction.png")
base_img.save(output_path)

print(f"💾 Saved annotated frame to: {output_path}")
print("✅ Inference completed successfully")
