# OMU-MAE — how to run the full pipeline

## ▶ Which notebook
Run **`omumae_full_pipeline.ipynb`** (the all-in-one: KITTI pretrain → linear probe →
fine-tune → qualitative figure → nuScenes cross-sensor transfer).

- `kitti_pretrain_omumae_full.ipynb` = KITTI-only component (legacy; not needed).
- `*.bak` = backups. **Do not run these.**

## Hardware
- **RTX 5090 (32 GB)**, RunPod Community Cloud, ~$0.99/hr. (H100 = ~3.3× cost for ~2× speed on this
  data-bound job — not worth it.)
- **400 GB network volume** mounted at `/workspace` (persists across pod restarts; cheaper + safer
  than the pod-local volume disk). KITTI (~100–150 GB) + nuScenes trainval keyframes (~70 GB) +
  ViT-B DINOv2 cache (~80 GB) + outputs.

## Credentials — ENV VARS ONLY, never in the notebook/repo
Use freshly **rotated** keys.
```bash
export KAGGLE_USERNAME=...   KAGGLE_KEY=...
mkdir -p ~/.kaggle && printf '{"username":"...","key":"..."}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
# nuScenes full trainval (only needed when SMOKE=False) — for the community downloader:
export NUSC_EMAIL=...   NUSC_PASSWORD=...
```

## Step 1 — SMOKE dry run (~20–40 min)  ← do this first
Top of the notebook has `SMOKE = True`. Leave it. Then **Run All** (Jupyter) or:
```bash
pip install -q kagglehub nuscenes-devkit pyquaternion scikit-image
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 omumae_full_pipeline.ipynb
```
**Pass = ** no OOM, KITTI + fine-tune + nuScenes-mini tables print, `results/` fills with JSON + PNG.
Note the pretrain **s/step** to estimate the full-run time.

## Step 2 — Full publication run (~2–4 days)
1. Stage **nuScenes v1.0-trainval keyframes** on the volume so `data/nuscenes/` has
   `samples/  maps/  v1.0-trainval/  lidarseg/v1.0-trainval/` (community downloader:
   github.com/li-xl/nuscenes-download, using the `NUSC_*` env vars).
2. Set **`SMOKE = False`**.
3. Re-run all (same nbconvert command).

## Config (already set for publication)
- Voxel grid 128×128×32 @ 0.4 m · frozen **DINOv2 ViT-B/14** target
- 5 variants: random / occmae / nomask / **full (OMU-MAE)** / **slidr (SLidR re-impl)**
- KITTI: full data, 20k pretrain steps, official SemanticKITTI split (**val = seq 08**)
- nuScenes: official split, all 6 cameras, frozen transfer probe, mean±std over 3 seeds
- ViT-L knob: set `dinov2_model='dinov2_vitl14'` + `image_feat_dim=1024` (heavier; batch 2–3)

## Results
- `results/final_results.json` — KITTI linear probe mIoU
- `results/finetune_results.json` — KITTI fine-tune (**main label-efficiency result**)
- `results/nuscenes_transfer_results.json` — cross-sensor transfer (mean±std)
- `results/*.png` — label-efficiency, 4-way comparison, qualitative example
- Expect: `full > slidr ≈ nomask > occmae > random`

## After the run
Pull `results/` + checkpoints to your machine, push to GitHub for reference,
then **terminate the pod and delete the network volume** to stop storage charges.

## Security
The notebook reads creds from env vars; they are never written to disk. If you ever pasted creds
anywhere, **rotate them**. `.gitignore` keeps `data/`, `*.bak`, `*.pt`, `kaggle.json`, tokens out of git.
