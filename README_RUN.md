# OMU-MAE — Full Pipeline Run & Results Guide

End-to-end instructions to run the whole pipeline on a cloud GPU and save the results.
Read top to bottom. Don't skip the SMOKE step.

---

## 0. TL;DR
1. RunPod **RTX 5090**, **400 GB network volume**, template `runpod-torch`, Jupyter on.
2. `git clone https://github.com/badrmellal/full_omu_mae.git`
3. Set **Kaggle** env creds (rotated).
4. Run **`omumae_full_pipeline.ipynb`** with `SMOKE = True` → ~30 min dry run → confirm green.
5. Stage **nuScenes trainval keyframes**, set `SMOKE = False`, run again → ~2–4 days.
6. Push `results/` back to this repo. Terminate pod + delete volume.

---

## 1. Which notebook
Run **`omumae_full_pipeline.ipynb`** — the all-in-one:
- **Part 1 (KITTI):** pretrain 5 variants → linear probe → fine-tune (main result) → qualitative figure.
- **Part 2 (nuScenes):** cross-sensor frozen-probe transfer, reusing Part 1's checkpoints.

Ignore `kitti_pretrain_omumae_full.ipynb` (KITTI-only component) and any `*.bak` (backups).

---

## 2. Hardware (RunPod)
- **GPU: RTX 5090 (32 GB)**, Community Cloud, ~$0.99/hr. Do **not** use H100 — this job is data-bound +
  small model, so H100 costs ~3.3× for ~2× speed (worse $/result).
- **Storage: 400 GB Network volume** mounted at `/workspace` (persists across pod stop/terminate;
  bills ~$0.07/GB/mo separately from compute). Disk budget:
  - KITTI (Kaggle, full) ~100–150 GB · nuScenes trainval keyframes ~70 GB · DINOv2 ViT-B cache ~80 GB · outputs ~10 GB.
- Estimated cost of one full run: **~$50–100 compute + ~$5–15 storage**.

---

## 3. Get the code on the pod
```bash
cd /workspace
git clone https://github.com/badrmellal/full_omu_mae.git
cd full_omu_mae
git config user.name  "Badr Mellal"
git config user.email "bandyscars@gmail.com"
```

---

## 4. Credentials — ENV VARS ONLY (rotated keys, never commit them)
```bash
# Kaggle (required — KITTI + SemanticKITTI auto-download):
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_rotated_key
mkdir -p ~/.kaggle
printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# nuScenes (only for the FULL run, when SMOKE=False):
export NUSC_EMAIL=your_email
export NUSC_PASSWORD=your_rotated_password
```
These are read at runtime; the notebook never writes them to disk. `.gitignore` blocks `kaggle.json`/`*.token`/`.env`.

---

## 5. Install dependencies
The `runpod-torch` template already has PyTorch + CUDA. Add:
```bash
pip install -q kagglehub nuscenes-devkit pyquaternion scikit-image timm
```
(The notebook also auto-installs these if missing.)

---

## 6. SMOKE dry-run — DO THIS FIRST (~20–40 min)
`SMOKE = True` is already set at the top of the notebook (cell right after CFG). It shrinks everything:
400 KITTI frames, 200 steps, 1 epoch, 1 label fraction, nuScenes-mini (auto public download), 1 seed.

Run it:
- **Jupyter:** open `omumae_full_pipeline.ipynb` → Run All. **or**
- **Headless:**
  ```bash
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 omumae_full_pipeline.ipynb
  ```

**Pass criteria (all must hold before the full run):**
- No OOM / no errors to the last cell.
- KITTI probe table, fine-tune table, and nuScenes-mini transfer table all print.
- `results/` fills with `.json` + `.png`.
- Note the pretrain **seconds/step** from the progress bar → multiply by 20,000 × 4 variants to sanity-check full-run time.

If OOM: lower batch in the pod-tuning cell (`CFG['train']['batch_size'] = 2`).

---

## 7. Full publication run (~2–4 days)
1. **Stage nuScenes v1.0-trainval keyframes** onto the volume (one-time). Using your **rotated** account:
   ```bash
   pip install nuscenes-devkit
   git clone https://github.com/li-xl/nuscenes-download
   # set NUSC_EMAIL / NUSC_PASSWORD env vars, download the v1.0-trainval KEYFRAME blobs
   # + nuScenes-lidarseg-all into  /workspace/full_omu_mae/data/nuscenes/
   ```
   Target layout (keyframes only is enough — skip `sweeps/` to save ~230 GB):
   ```
   data/nuscenes/
     samples/   maps/   v1.0-trainval/   lidarseg/v1.0-trainval/
   ```
2. **Edit the notebook:** set `SMOKE = False`.
3. **Run all** (same nbconvert command as above, or Jupyter Run All). Use `nohup`/`tmux` so an SSH drop
   doesn't kill it:
   ```bash
   nohup jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=-1 omumae_full_pipeline.ipynb > run.log 2>&1 &
   tail -f run.log
   ```

If nuScenes trainval is **not** staged, Part 2 prints an ACTION-NEEDED message and **skips gracefully** —
the KITTI results (Part 1) still complete and save.

---

## 8. Where results are saved
The last cell auto-collects everything into **`results/`**:
| File | What |
|------|------|
| `results/final_results.json` | KITTI linear-probe mIoU (5 variants × label fractions) |
| `results/finetune_results.json` | KITTI fine-tune mIoU — **main label-efficiency result** |
| `results/nuscenes_transfer_results.json` | nuScenes cross-sensor transfer (mean ± std over seeds) |
| `results/finetune_label_efficiency.png` | fine-tune curves |
| `results/probe_comparison_4way.png` | probe bar chart |
| `results/qualitative_example.png` | real-sample figure |

Checkpoints (large) live in `data/runs/kitti_omumae_full/<variant>/ckpt.pt` (git-ignored).
Expected ordering in the tables: **full ≳ slidr ≈ nomask > occmae > random**.

---

## 9. Save results back to GitHub
The pod has no access to your laptop keychain, so pushing needs a **GitHub PAT**
(create a fine-grained token with write access to `full_omu_mae`, use it, then revoke).

```bash
cd /workspace/full_omu_mae
git add results/ omumae_full_pipeline.ipynb        # executed notebook keeps its output cells
git commit -m "Add executed results (5-variant ablation + SLidR + KITTI fine-tune + nuScenes transfer)"
git push https://<YOUR_PAT>@github.com/badrmellal/full_omu_mae.git main
```

**Checkpoints** (~28 MB each) are git-ignored on purpose. To keep them for reference, attach to a
GitHub Release instead of the repo:
```bash
# needs gh + auth, or upload via the GitHub web UI under Releases
gh release create v1-omumae data/runs/kitti_omumae_full/*/ckpt.pt -t "OMU-MAE checkpoints"
```
GitHub renders the executed `.ipynb` (figures + tables) directly — it's a viewable reference with no Colab needed.

---

## 10. Teardown (stop the bleeding $$)
1. Pull/confirm results are pushed to GitHub.
2. **Terminate the pod** (stops the $0.99/hr immediately).
3. **Delete the network volume** if you won't rerun (stops the ~$0.07/GB/mo).

---

## 11. Config summary (already set for publication)
- Voxel grid **128×128×32 @ 0.4 m** (dense 3D CNN; finer needs sparse conv — out of scope).
- Frozen **DINOv2 ViT-B/14** target (768-d). ViT-L knob: set `dinov2_model='dinov2_vitl14'` +
  `image_feat_dim=1024` (heavier; batch 2–3).
- 5 variants: `random / occmae (Occupancy-MAE) / nomask (CleverDistiller) / full (OMU-MAE) / slidr (SLidR re-impl)`.
- KITTI: full data, 20k pretrain steps, **official SemanticKITTI split (val = sequence 08)**.
- nuScenes: official split, **all 6 surround cameras** (Velodyne HDL-32E), frozen transfer probe,
  **mean ± std over 3 seeds**.
- CUDA: bf16 autocast (via `amp_ctx`), VFM-aware batch (4 for ViT-B on 32 GB).

---

## 12. Troubleshooting
| Symptom | Fix |
|--------|-----|
| `kaggle` auth error | set `KAGGLE_USERNAME`/`KAGGLE_KEY` + `~/.kaggle/kaggle.json` (chmod 600) |
| CUDA out of memory | lower `CFG['train']['batch_size']` to 2 in the pod-tuning cell |
| nuScenes Part skipped | data not staged at `data/nuscenes/` (full run) — see §7; KITTI still completes |
| `wget` 403 on nuScenes trainval | trainval needs login; stage via the community downloader (§7). Mini is public. |
| SSH dropped, run died | use `nohup`/`tmux` (§7) |
| push asks for password | GitHub needs a PAT, not your password (§9) |

---

## 13. Security & double-blind
- All credentials are env vars; nothing is written to disk or committed. **Rotate any key you've ever pasted.**
- WACV / IEEE IV are **double-blind**: a public repo under your name during review is a de-anonymization
  risk. Keep `full_omu_mae` **private** until acceptance (GitHub → Settings → Change visibility),
  or use an anonymized mirror (`anonymous.4open.science`) for the submission's code link.
