# OMU-MAE — paper sources (IEEE IV + WACV)

One body, two venue wrappers. The text/figures/tables are written **once**
in `sections/` + `refs.bib`; each venue gets a thin `main_*.tex` that sets
the class, title block, and bibliography style.

## Does the format change between venues? — Yes.

| | **IEEE IV** | **WACV** |
|---|---|---|
| Class | `IEEEtran` (conference), 2-col, 10pt | CVF `wacv` over `article`, 2-col, 10pt |
| Build file | `main_ieee.tex` | `main_wacv.tex` |
| Bib style | `IEEEtran.bst` | `ieee_fullname.bst` |
| Page limit | ~**6 pp** + refs (confirm on the IV 2027 CFP; +2 pp usually allowed for a fee) | **8 pp** + unlimited refs (hard desk-reject over 8) |
| Review | double-blind | double-blind (`[review]` option auto-anonymizes + adds line numbers) |
| Keywords | `\begin{IEEEkeywords}` | none |

Same science, different template + different page budget. You cannot
submit one `.tex` to both — but you maintain one `sections/` folder and
flip the wrapper.

> Do **not** submit to both venues at once (dual-submission ban). Plan:
> WACV 2027 first (earlier deadline); if rejected, IEEE IV 2027.

## Structure
```
paper/
  main_ieee.tex      <- IEEE IV build
  main_wacv.tex      <- WACV build
  refs.bib           <- shared bibliography
  sections/          <- shared body (\input by both mains)
    00_abstract.tex 01_intro.tex 02_related.tex 03_method.tex
    04_setup.tex 05_results.tex 06_discussion.tex
    07_limitations.tex 08_conclusion.tex
    fig_pipeline.tex <- TikZ architecture diagram (Fig 1)
  figures/           <- result PNGs (from the RunPod run)
```

## How to build (Overleaf — no local TeX needed)

**IEEE IV:** New Project → Upload `paper/` → set **`main_ieee.tex`** as the
main document → Recompile. `IEEEtran.cls`/`.bst` ship with Overleaf.

**WACV:** the CVF class isn't bundled. Either:
1. Open the official **WACV 2027 Author Kit** (or the "WACV Author Kit"
   Overleaf template), grab `wacv.sty` + `ieee_fullname.bst`, drop them
   next to `main_wacv.tex`; set `main_wacv.tex` as main; Recompile. **or**
2. Start *from* the kit project and replace its body with our
   `\input{sections/...}` lines (preamble/class stay from the kit — safest,
   guarantees the year's exact macros).

## Two switches (top of each main)

- `\anontrue` / `\anonfalse` — double-blind. **Keep `\anontrue` for
  submission.** Flip to `\anonfalse` (and WACV `[review]`→`[final]`) for the
  camera-ready / arXiv version with author names + the GitHub link.
- `\extrafigstrue` / `\extrafigsfalse` — optional figures (pretraining
  dynamics, reconstruction, confusion matrices). **WACV = on, IEEE = off**
  (defaults set) to respect the 6 vs 8 page budgets. If WACV runs over 8 pp,
  set it `false`; if you still overflow, the per-class figure (`fig:perclass`)
  is the next safe cut.

## Figures
Used: `fig_pipeline` (TikZ), `probe_comparison_5way`, `probe_per_class_5way`,
`nuscenes_transfer_5way` (always); `loss_curves`, `reconstruction_full`,
`probe_confusion_4way` (only when `\extrafigs`). Also copied and available if
you want them: `qualitative_example`, `retrieval_full`.

## Numbers: what's real vs. what to fill
- **Real (from your RunPod run, ViT-B/14, single seed):** the linear-probe
  table (`tab:probe`), the nuScenes transfer table (`tab:nusc`), the
  retrieval diagnostic numbers, all per-class statements. These match
  `results/final_results.json` + `nuscenes_transfer_results.json`.
- **Qualitative only:** §5.1 dynamics prose ("converges above 97%…"). If you
  keep §5.1, drop the exact loss values from your training logs into the
  caption/text; I avoided inventing precise curve values for the ViT-B run.
- **Pilot, flagged:** the semantics-driven-masking negative result
  (Limitations §6) used the older **ViT-S/14** backbone. It's written as a
  pilot; re-run at ViT-B/14 before promoting it to a table.

## Before you submit
1. **Rotate the credentials** you pasted in chat (Kaggle key + token,
   nuScenes password, GitHub PAT). They're compromised. None are in this repo.
2. Confirm the **IV 2027** page limit + double-blind policy on the official
   CFP when it posts (2027 page hasn't been finalized; we used IV's standard
   6 pp / double-blind).
3. If compute allows, add **seeds 1–2** (the run is resumable) so the small
   in-domain deltas and the nuScenes numbers get error bars — the single seed
   is the #1 reviewer target.
4. Camera-ready: `\anonfalse`, restore the GitHub link, add acknowledgments.
```
