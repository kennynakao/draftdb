# 11 — Proven signals, their proxies, and the tests they imply (2026-10-21)

Scope: what the literature and this project's own screens agree predicts NBA value beyond the draft slot; which of those signals the inputs already carry; what correlates with them that the engine has not been given; and the tests run from it (queue R on the A100, `experiments/lab_2026-10-19/gpu_r/`).

## What is proven (literature catalog in 01/02 plus the project's 2000–2011 screen)

| signal | evidence | in the inputs? |
|---|---|---|
| age at draft (younger better, about 0.5 WARP per year per Pelton) | Pelton, Vashro, every public model; project screen −0.19 | yes, monotone constraint |
| scout / mock consensus rank | ESPN S&I: "by far the most important variable"; CARMELO | yes (dated, 2–6 publishers; more publishers tested 2026-10-20, no change) |
| production (BPM-type impact), first-season BPM in particular | project screen +0.31 for first-season college BPM | yes (col_impact, col_first_bpm, tc_ age-relative) |
| steal rate, especially for guards; offensive rebounding for bigs | ESPN S&I; bestballstats; project screen +0.26 | levels yes; the position interaction no |
| assists, assist/turnover | Engelmann, LUCARIO; project screen +0.21 | levels yes; the ratio no |
| FT% translates (R² ≈ 0.53 college→NBA), 3P% barely (≈ 0.08) | Wharton project, Pelton, Vashro | both levels yes; shrinkage treats them alike (M = 400) |
| height / length composite | Teramoto 2018; Engelmann | yes (dv_ length composite tested, failed as a block) |
| international status | Engelmann (large positive) | implicit through coverage; no explicit flag |
| intangibles / leadership grades | Johnson 2016 (~11% of variance) | yes (sc_, in the champion) |
| combine athletic tests | little signal (Teramoto, PRISM) | present, low weight |

Nothing published in 2025–2026 adds a predictor to this list: the 2026 public models (Funnyball on Engelmann's data, CraftedNBA, SI's "productive young athlete" screen, Dynasty NBA) recombine age, production, athleticism and consensus.

## Correlates and re-specifications the engine had not been given

1. **Differential shrinkage** (`shrink_profile`): 3P% shrunk hard (M = 1500) toward an FT%-implied prior fitted on the training rows (3P% ≈ 0.20 + 0.21·FT% among players with 2+ attempts per 40), steals, blocks and ORB% barely shrunk (M = 100), everything else as before. Mechanism: the most and least stable translators should not share one reliability curve.
2. **Interaction terms from the public models** (`lucario`): assist/turnover ratio, body density (weight / height³), assists × rebounds, steal rate × guard, ORB% × big, block rate × big, FT% × youth. Trees can find these; the ridge and the in-context model cannot unless given them.
3. **Label-reliability weights** (`hzw`): a training row whose label rests on k seasons is weighted by the k-season-to-five-season rank agreement measured on the complete classes (0.60, 0.82, 0.91, 0.97, 1.0), to the power hzw. Mechanism: the engine's season-consistent folds and the expanding window train on one- and two-season labels as if they were final.
4. **Projected labels**: scaling a truncated value to its expected five-season value is a per-class linear map and is invisible to a within-class rank label; dropped after the check (correlation 1.000 with the plain label).

Not built: teammate pre-draft consensus sums (Kuehn 2023; needs roster-to-pid joins, teammate BPM proxies exist in tc_/f50_); an explicit international flag (trivial, queued for the next round if anything above moves).
