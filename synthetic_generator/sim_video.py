#!/usr/bin/env python3
"""
Physics-based synthetic fertilizer distribution simulator.

Generates:
- Top-view synthetic video
- Landing point scatter plot
- Histogram distribution CSV
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio


# ======================================================
# ARGUMENTS
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic fertilizer spread simulator")

    parser.add_argument("--output_dir", type=str, default="simulation_outputs")
    parser.add_argument("--total_time", type=float, default=5.0)
    parser.add_argument("--frequency", type=int, default=250)
    parser.add_argument("--disk_radius", type=float, default=0.55)
    parser.add_argument("--angular_speed", type=float, default=94.25)
    parser.add_argument("--cart_speed", type=float, default=-1.11)
    parser.add_argument("--n_blades", type=int, default=4)
    parser.add_argument("--frame_size_m", type=float, default=60.0)

    return parser.parse_args()


# ======================================================
# PARTICLE CLASS
# ======================================================
class Particle:
    def __init__(self, position, velocity):
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.alive = True

    def update(self, dt, g, drag_coefficient):
        if not self.alive:
            return

        v = self.vel
        drag_acc = -drag_coefficient * v * np.linalg.norm(v)
        acc = np.array([0, 0, -g]) + drag_acc

        self.vel += acc * dt
        self.pos += self.vel * dt

        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.alive = False


# ======================================================
# MAIN SIMULATION
# ======================================================
def main():

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Physics constants
    g = 9.81
    drag_coefficient = 0.07
    particle_mass = 0.01

    dt = 1 / args.frequency / 2
    n_steps = int(args.total_time / dt)
    release_interval = int(1 / args.frequency / dt)

    particles = []
    landing_positions = []

    disk_configs = [(-0.555, 0, 1, 1), (0.555, 0, 1, -1)]

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(-args.frame_size_m / 2, args.frame_size_m / 2)
    ax.set_ylim(-args.frame_size_m / 2, args.frame_size_m / 2)
    ax.set_xticks([])
    ax.set_yticks([])

    frames = []

    # ===============================
    # SIMULATION LOOP
    # ===============================
    for step in tqdm(range(n_steps), desc="Simulating"):
        t = step * dt
        cart_y = args.cart_speed * t

        if step % release_interval == 0:
            for disk_x, disk_y, disk_z, direction in disk_configs:
                for blade_index in range(args.n_blades):

                    angle = (args.angular_speed * t +
                             blade_index * (2 * np.pi / args.n_blades))

                    tangential = args.angular_speed * args.disk_radius * np.array([
                        -np.sin(angle),
                        np.cos(angle),
                        0
                    ])

                    position = [disk_x, cart_y, disk_z]
                    velocity = np.array([0, args.cart_speed, 0]) + tangential

                    particles.append(Particle(position, velocity))

        for p in particles:
            prev_alive = p.alive
            p.update(dt, g, drag_coefficient)

            if prev_alive and not p.alive:
                landing_positions.append((p.pos[0], p.pos[1]))

        # Capture frame
        alive_positions = np.array([p.pos for p in particles if p.alive])

        if len(alive_positions) > 0:
            ax.clear()
            ax.set_facecolor("black")
            ax.set_xlim(-args.frame_size_m / 2, args.frame_size_m / 2)
            ax.set_ylim(-args.frame_size_m / 2, args.frame_size_m / 2)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.scatter(alive_positions[:, 0],
                       alive_positions[:, 1] - cart_y,
                       s=5, color="white")

            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(frame)

    plt.close(fig)

    # ======================================================
    # SAVE VIDEO
    # ======================================================
    video_path = os.path.join(args.output_dir, "synthetic_video.mp4")
    writer = imageio.get_writer(video_path,
                                fps=int(1 / (dt * 10)),
                                format="ffmpeg")

    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"✅ Video saved: {video_path}")

    # ======================================================
    # SAVE LANDING DISTRIBUTION
    # ======================================================
    landing_positions = np.array(landing_positions)

    plt.figure(figsize=(8, 5))
    plt.scatter(landing_positions[:, 0],
                landing_positions[:, 1],
                alpha=0.7, s=10)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)

    scatter_path = os.path.join(args.output_dir, "landing_scatter.png")
    plt.savefig(scatter_path)
    plt.close()

    print(f"✅ Scatter saved: {scatter_path}")

    # ======================================================
    # HISTOGRAM
    # ======================================================
    counts, bins = np.histogram(landing_positions[:, 0], bins=50)

    plt.figure(figsize=(8, 4))
    plt.hist(landing_positions[:, 0], bins=50)
    plt.xlabel("X position (m)")
    plt.ylabel("Count")

    hist_path = os.path.join(args.output_dir, "histogram.png")
    plt.savefig(hist_path)
    plt.close()

    print(f"✅ Histogram saved: {hist_path}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "Bin_Start": bins[:-1],
        "Bin_End": bins[1:],
        "Count": counts
    })

    csv_path = os.path.join(args.output_dir, "histogram_bins.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ CSV saved: {csv_path}")
    print("🎯 Simulation completed successfully")


if __name__ == "__main__":
    main()



