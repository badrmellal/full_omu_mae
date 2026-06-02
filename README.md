# OMU-MAE

**Cross-Modal Masked Voxel Pretraining with Frozen Vision Foundation Targets for Autonomous Driving Perception**

Badr Mellal, Rabab Benfouina, Ahmed Drissi el Maliani — LRIT Laboratory, Faculty of Sciences in Rabat, Mohammed V University in Rabat, Morocco.

> 📄 Paper (LaTeX, IEEE IV + WACV builds): [`paper/`](paper/) · see [`paper/README.md`](paper/README.md) for build instructions.

---

## Overview

OMU-MAE (**O**ccupancy + **M**ulti-modal + **U**nified Masked Autoencoder) is a voxel-level masked autoencoder for self-supervised pretraining on paired camera + LiDAR data.

Each scene is voxelized at 0.4 m into a 128×128×32 grid. Per-voxel features are populated by projecting **frozen DINOv2 ViT-B/14** (768-d) patch features through the camera–LiDAR calibration. Voxels are masked at high rate with the **range-aware schedule** of Occupancy-MAE, and a 3D CNN encoder–decoder reconstructs:

1. Per-voxel binary **occupancy** (BCE), and
2. The masked **DINOv2 features** at occupied positions (MSE).

OMU-MAE fills the empty cell of the cross-modal SSL design space — *masked reconstruction with a frozen VFM as target* — and is evaluated head-to-head against Occupancy-MAE, a re-implemented SLidR, and a no-mask (CleverDistiller-equivalent) baseline.

| Target \ Objective | Distillation                         | Masked reconstruction       |
|--------------------|--------------------------------------|-----------------------------|
| Raw modality       | —                                    | UniM²AE, NS-MAE             |
| Frozen VFM         | SLidR, ScaLR, CleverDistiller        | **OMU-MAE (this work)**     |

---

## Headline result — SemanticKITTI linear probe

Frozen-encoder linear probe (single 1×1×1 conv head), 19-class voxel mIoU (%). Same architecture + probe protocol for all five conditions; only the pretraining recipe differs.

| Pretraining variant                          | 1%    | 5%    | 10%   | 100%  | Δ@100 |
|-----------------------------------------------|-------|-------|-------|-------|-------|
| Random init                                   | 4.95  | 5.72  | 6.14  | 5.50  | −14.19 |
| Occupancy-MAE (LiDAR-only re-impl.)           | 13.88 | 16.34 | 17.08 | 17.91 | −1.77 |
| SLidR (contrastive-VFM re-impl.)              | 12.40 | 13.82 | 14.66 | 15.46 | −4.22 |
| No-mask (CleverDistiller-equivalent)          | 9.75  | 12.26 | 14.18 | 16.41 | −3.27 |
| **OMU-MAE (ours)**                            | **14.92** | **17.49** | **18.56** | **19.68** | — |

**OMU-MAE wins at every label fraction.** The three controlled deltas at 100% labels isolate each design choice: **+1.77** vs Occupancy-MAE (the cross-modal target), **+3.27** vs no-mask (the masking inductive bias), **+4.22** vs SLidR (masked reconstruction vs contrastive distillation of the *same* frozen-VFM target).

## Cross-sensor transfer — nuScenes (frozen probe, single seed)

| Pretraining variant | 1% | 5% | 10% | 100% |
|---------------------|----|----|-----|------|
| Random init         | 0.45 | 0.58 | 0.68 | 0.83 |
| Occupancy-MAE       | 0.30 | 0.37 | 0.46 | 0.61 |
| SLidR               | 0.28 | 0.60 | 0.74 | 0.94 |
| No-mask (CleverDist.-eq.) | **1.06** | **1.54** | **1.71** | **2.45** |
| OMU-MAE (ours)      | 0.76 | 1.41 | 1.49 | 1.62 |

Reported honestly: absolute mIoU is low (a 64-beam KITTI encoder, frozen, probed on a 32-beam nuScenes sensor; single seed). The robust pattern is that the **two cross-modal DINOv2-target variants (OMU-MAE + no-mask) separate from the LiDAR-only / contrastive / random baselines — the cross-modal target is what transfers**, while masked-vs-no-mask is within single-seed noise (the masking benefit is in-domain-specific).

---

## Method

```
Camera RGB  ──► DINOv2 ViT-B/14 (frozen) ──► 16×16×768 patch features ──┐
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

- **Cross-modal voxel input.** Occupancy O ∈ {0,1}^(X×Y×Z) + per-voxel mean-pooled DINOv2 feature volume Fv ∈ R^(768×X×Y×Z), concatenated with a binary mask indicator → input (1 + 768 + 1) × X × Y × Z.
- **Range-aware masking.** Pr[m = 1] = ρ·(1 + α·d)⁻¹, ρ = 0.85, α = 0.5; near-field (dense LiDAR) masked more.
- **Encoder.** 4-stage 3D CNN (Conv3D + GroupNorm + GELU), strided → bottleneck 256×32×32×8.
- **Decoder.** Mirror with trilinear upsampling; two heads (occupancy logit + 768-d feature prediction).
- **Loss.** `L = 1.0·L_occ + 0.5·L_feat`, both at masked positions (feature loss at masked ∩ occupied).

---

## Repo layout

```
kitti_omu_mae/
├── README.md                    # this file
├── paper/                       # LaTeX (IEEE IV + WACV) — see paper/README.md
├── omumae_full_pipeline.ipynb   # main end-to-end notebook (pretrain → probe → nuScenes transfer)
├── results/                     # result JSONs + figures (the numbers above)
├── docs/
│   └── README_RUN.md            # detailed cloud-GPU run guide
└── legacy/
    ├── kitti_pretrain_omumae_full.ipynb   # earlier KITTI-only notebook (superseded)
    └── run_nuscenes.py                    # standalone nuScenes re-run (notebook Part 2 supersedes it)
```

The notebook trains and probes **five variants** in one run (`random` / `occmae` / `slidr` / `nomask` / `full`), then runs the nuScenes cross-sensor transfer (Part 2). The frozen linear probe is the primary result; end-to-end fine-tuning is left as future work and is not included in this submission.

---

## Setup & reproduce

**Data (Kaggle):** `hocop1/kitti-odometry` (left-camera + Velodyne + calib) and `luischavarriazamora/semantic-kitti` (per-point labels). nuScenes (`v1.0-trainval`) needs a nuScenes account.

**Backbone:** frozen DINOv2 ViT-B/14 (768-d) via `torch.hub` (config in cell 7).

**Hyperparameters (pretraining):** AdamW, lr 5e-4, weight decay 0.05, batch 4, 20,000 steps, grad-clip 1.0, mask ρ=0.85, range decay α=0.5, 16,384 LiDAR pts/scene, focal occupancy (α=0.25, γ=2.0).

**Hardware:** runs on a single GPU (publication run: NVIDIA RTX PRO 6000); device-portable (CUDA bf16 / MPS fp32 / CPU fallback).

1. Open `omumae_full_pipeline.ipynb` (Colab / RunPod / local). Set Kaggle creds via env vars (`KAGGLE_USERNAME`, `KAGGLE_KEY`) — **never hard-code credentials**.
2. Run the Setup cell (downloads KITTI + SemanticKITTI), then run all cells.
3. Artifacts (JSONs + figures) collect into `results/`. See `docs/README_RUN.md` for the detailed cloud-GPU run guide.

---

## Key findings

- **Pretraining matters.** All four pretrained variants beat random init at every label fraction.
- **Cross-modal target helps.** OMU-MAE > Occupancy-MAE by +1.77 pp @100% (the DINOv2 target).
- **Masking helps in-domain.** OMU-MAE > no-mask by +3.27 pp @100% (and more in the low-label regime, +5.17 pp @1%).
- **Masked reconstruction > contrastive distillation.** OMU-MAE > re-implemented SLidR by +4.22 pp @100%, for the same frozen-VFM target.
- **Cross-sensor:** the cross-modal target transfers (OMU-MAE + no-mask lead on nuScenes); the masking benefit is in-domain-specific.
- **Preliminary negative result (ViT-S pilot):** DINOv2-feature-norm-guided ("semantic") masking underperforms range-aware masking by 3–5 pp; in 3D the right principle is "mask the sensor-dense," not "mask the semantically rich."

## Limitations

1. **Single seed** — point estimates, no error bars (the run is resumable; adding seeds is the main next step).
2. Occupancy-MAE / SLidR / no-mask are faithful **re-implementations** in our dense 3D CNN framework, not the authors' original code.
3. **Cross-sensor transfer is preliminary** (frozen probe, single seed, low absolute mIoU).
4. **End-to-end fine-tuning** is future work (the frozen linear probe is the primary representation-quality measure here).
5. The 0.4 m voxel grid is coarse for small classes (motorcycle/person/bicyclist ≈ 0 mIoU under all conditions).

---

## Citation

```bibtex
@misc{mellal2026omumae,
  title  = {OMU-MAE: Cross-Modal Masked Voxel Pretraining with Frozen Vision Foundation Targets for Autonomous Driving Perception},
  author = {Mellal, Badr and Benfouina, Rabab and Drissi el Maliani, Ahmed},
  year   = {2026},
  note   = {LRIT Laboratory, Mohammed V University in Rabat}
}
```

## Contact
badr_mellal@um5.ac.ma · r.benfouina@um5r.ac.ma · a.elmaliani@um5r.ac.ma
