# Swath Width Prediction using CNN–Transformer

Hybrid CNN–Transformer model for real-time swath width estimation in centrifugal fertilizer spreading.

This repository implements the methodology described in our study on temporal modeling of fertilizer distribution patterns using synthetic video sequences under a Sim2Real paradigm.


## 🔍 Overview

The pipeline consists of:

1. **Synthetic video generation** based on physics-driven projectile motion.
2. **Frame-level feature extraction** using a CNN backbone (ResNet18 or EfficientNet).
3. **Temporal modeling** using a Transformer encoder.
4. **Regression head** for predicting:
   - Scalar swath width (meters)
  

---

Example to use scripts:


python synthetic_generator/sim_video.py \
    --output_dir synthetic_outputs \
    --angular_speed 94.25 \
    --cart_speed -1.11




python training/train.py \
    --data_root dataset/synthetic/wide_FOV \
    --labels_csv dataset/synthetic/wide_FOV/labels.csv \
    --num_frames 25 \
    --output_dir outputs



python training/test.py \
    --test_data test_data.pt \
     --weights wide_fov_best_model.pth 
    --output_dir outputs


    


python preprocessing/preprocess.py \
    --in_dir dataset/real/Experiment1/raw_frames \
    --out_dir  dataset/real/Experiment1/processed_frames \
    --resize_long 2208 \
    --square_pad 2208 \
    --clahe \
    --threshold otsu


python inference/predict_synthetic.py \
    --frames_dir sample_sequence \
    --weights wide_fov_best_model.pth



python inference/predict_real.py \
    --frames_dir dataset/real/Experiment1/processed_frames \
    --weights wide_fov_best_model.pth    



   



