# OMU-MAE

**Cross-Modal Masked Voxel Pretraining with Frozen Vision Foundation Targets for Autonomous Driving Perception**

Badr Mellal, Rabab Benfouina, Ahmed Drissi el Maliani — LRIT Laboratory, Faculty of Sciences in Rabat, Mohammed V University in Rabat, Morocco.

---

## Overview

OMU-MAE (**O**ccupancy + **M**ulti-modal + **U**nified Masked Autoencoder) is a voxel-level masked autoencoder for self-supervised pretraining on paired camera + LiDAR data.

Each scene is voxelized at 0.4 m resolution into a 128×128×32 grid. Per-voxel features are populated by projecting **frozen DINOv2 ViT-S/14** patch features through the camera–LiDAR calibration. Voxels are masked at high rate using the **range-aware schedule** of Occupancy-MAE, and a 3D CNN encoder–decoder is trained to reconstruct:

1. Per-voxel binary **occupancy** (BCE).
2. Masked **DINOv2 features** at occupied positions (MSE).

OMU-MAE fills the empty cell of the cross-modal SSL design space — *masked reconstruction with a frozen VFM as target* — and is positioned explicitly against Occupancy-MAE, UniM²AE, NS-MAE, SLidR, ScaLR, and CleverDistiller.

| Target \ Objective | Distillation                         | Masked reconstruction       |
|--------------------|--------------------------------------|-----------------------------|
| Raw modality       | —                                    | UniM²AE, NS-MAE             |
| Frozen VFM         | SLidR, ScaLR, CleverDistiller        | **OMU-MAE (this work)**     |

---

## Headline Results

Linear probe on SemanticKITTI (frozen encoder + 1×1×1 conv head), 19-class mIoU (%):

| Pretraining variant                       | 1%    | 5%    | 10%   | 100%  | Δ vs OMU-MAE |
|-------------------------------------------|-------|-------|-------|-------|--------------|
| Random init                               | 4.89  | 5.56  | 5.25  | 5.32  | −10.42       |
| Occupancy-MAE (LiDAR-only re-impl.)       | 8.66  | 11.55 | 11.96 | 14.31 | −1.43        |
| No-mask (CleverDistiller-equivalent)      | 9.24  | 11.01 | 11.46 | 14.24 | −1.50        |
| **OMU-MAE (ours)**                        | **10.78** | **13.20** | **14.13** | **15.74** | —    |

- Cross-modal target contribution: **+1.43 pp** at 100 % labels (vs Occupancy-MAE).
- Masking inductive bias contribution: **+1.50 pp** at 100 % labels (vs no-mask).
- Largest per-class gains: building **+36.6**, sidewalk **+32.8**, road **+25.1**, car **+23.0**, terrain **+21.4**, vegetation **+20.6** pp.

---

## Method

```
Camera RGB  ──► DINOv2 ViT-S/14 (frozen) ──► 16×16×384 patch features ──┐
                                                                        │ camera–LiDAR
                                                                        │ projection (P2, Tr)
                                                                        ▼
LiDAR points ─► voxelize (128×128×32) ──► O ⊕ Fv ──► range-aware mask (ρ=0.85)
                                                          │
                                                          ▼
                                      3D CNN encoder ─► bottleneck ─► 3D CNN decoder
                                                                          │
                                                  ┌───────────────────────┴────────────────────────┐
                                                  ▼                                                ▼
                                       Head: occupancy Ô (BCE)                  Head: feature F̂v (MSE)
                                       loss @ masked voxels                     loss @ masked ∩ occupied voxels
```

- **Cross-modal voxel input.** Binary occupancy O ∈ {0,1}^(X×Y×Z) + per-voxel mean-pooled DINOv2 feature volume Fv ∈ R^(384×X×Y×Z), concatenated with a binary mask indicator → input of shape (1 + 384 + 1) × X × Y × Z.
- **Range-aware masking.** Pr[m_xyz = 1] = ρ · (1 + α·d)^−1 with ρ = 0.85, α = 0.5; near-field voxels (dense LiDAR) are masked more aggressively.
- **Encoder.** 4-stage 3D CNN (Conv3D + GroupNorm + GELU) with strided downsampling → bottleneck H ∈ R^(256×32×32×8).
- **Decoder.** Mirror of encoder with trilinear upsampling, two heads (occupancy logit + 384-d feature prediction).
- **Loss.** `L = λ_occ · L_occ + λ_feat · L_feat` with λ_occ = 1.0, λ_feat = 0.5. `L_occ` evaluated at all masked voxels; `L_feat` at masked ∩ occupied voxels only.

---

## Repo Layout

```
kitti_omu_mae/
├── README.md
└── kitti_pretrain_omumae_full.ipynb     # end-to-end notebook (pretrain + probe)
```

The notebook trains and evaluates **four variants** in a single run:

| Variant   | Architecture                              | Pretraining objective                                                  |
|-----------|-------------------------------------------|------------------------------------------------------------------------|
| `random`  | OMU-MAE class (architecture-matched)      | none — fresh init, used as probe baseline                              |
| `occmae`  | `OccupancyMAEBaseline` (LiDAR-only 3D CNN) | range-aware mask + focal BCE on occupancy                              |
| `nomask`  | OMU-MAE class (unchanged)                 | `mask_ratio=0`; cross-modal feature regression at all positions (≈ CleverDistiller) |
| `full`    | OMU-MAE class (unchanged)                 | `mask=0.85` + dual reconstruction (current method)                     |

Probe protocol is identical across variants (frozen encoder + 1×1×1 conv probe on bottleneck features).

---

## Setup

### Data

The notebook downloads everything it needs from Kaggle:

- `hocop1/kitti-odometry` — 22 sequences of left-camera images, Velodyne LiDAR, calibration.
- `luischavarriazamora/semantic-kitti` — per-point class labels.

Pretraining uses KITTI odometry sequences 00–21 (8000 stratified frames, 90/10 train/val split). Evaluation uses SemanticKITTI sequences 00–10 with the official 19-class mapping.

### Hardware

Configured for a single consumer-grade GPU. Reference: NVIDIA RTX 4080 (16 GB VRAM); the notebook auto-detects MPS / CPU as fallback.

- Pretraining wall-clock: ≈ 1.5 h per variant.
- Probe wall-clock: ≈ 1 h per variant.
- Full 4-variant pretraining + 16 probe runs: ≈ 8 h end-to-end.

### Hyperparameters

| Stage         | Optimizer | LR        | Weight decay | Batch | Steps / epochs | Other                                |
|---------------|-----------|-----------|--------------|-------|----------------|--------------------------------------|
| Pretraining   | AdamW     | 5 × 10⁻⁴  | 0.05         | 2     | 5000 steps     | grad-clip 1.0, ρ = 0.85, α = 0.5, 16384 LiDAR pts/scene |
| Linear probe  | AdamW     | 5 × 10⁻³  | —            | 2     | 5 epochs       | cross-entropy with `ignore_index=0`  |

### Outputs

Each variant writes to `data/runs/kitti_omumae_full/{variant}/`. The final comparison plot and `final_results.json` land in `data/runs/kitti_omumae_full/`.

---

## How to Reproduce

1. Open `kitti_pretrain_omumae_full.ipynb` in Colab (T4/A100) or locally on a CUDA / MPS / CPU host.
2. Run the **Setup** cell (downloads KITTI + SemanticKITTI).
3. Run all cells — pretraining, probing, plotting, and `final_results.json` happen end-to-end.

---

## Key Findings

- **Pretraining matters.** All three pretrained variants outperform random init by a wide margin at every label fraction.
- **Cross-modal target helps.** OMU-MAE > Occupancy-MAE by +1.43 pp at 100 % labels (and +2.12 pp at 1 %); cross-modal DINOv2 supervision is most useful in the data-scarce regime.
- **Masking helps.** OMU-MAE > no-mask by +1.50 pp at 100 % labels and is consistent across all four label fractions (+1.54 / +2.19 / +2.67 / +1.50 pp).
- **Negative result on semantic-aware masking.** DINOv2-feature-norm-guided masking *underperforms* range-aware masking by 3–5 pp absolute mIoU on every one of the 19 classes. The 2D intuition “mask the semantically rich” does not transfer to voxel MAE; the right principle in 3D appears to be **“mask the sensor-dense.”**

---

## Limitations

1. The Occupancy-MAE and CleverDistiller-equivalent baselines are faithful re-implementations in our dense 3D CNN framework, not the authors’ original sparse-conv code.
2. Single-dataset evaluation (KITTI / SemanticKITTI). Transfer to nuScenes / Waymo Open is left for future work.
3. The 0.4 m voxel grid is coarse for small classes (motorcycle, person, bicyclist ≈ 0 mIoU under all conditions). A multi-resolution variant or sparse-voxel backbone with a shallow non-linear probe head would likely help fine-grained classes.

---

## Citation

```bibtex
@misc{mellal2025omumae,
  title  = {OMU-MAE: Cross-Modal Masked Voxel Pretraining with Frozen Vision Foundation Targets for Autonomous Driving Perception},
  author = {Mellal, Badr and Benfouina, Rabab and Drissi el Maliani, Ahmed},
  year   = {2025},
  note   = {LRIT Laboratory, Mohammed V University in Rabat}
}
```

## Contact

- badr_mellal@um5.ac.ma
- r.benfouina@um5r.ac.ma
- a.elmaliani@um5r.ac.ma
