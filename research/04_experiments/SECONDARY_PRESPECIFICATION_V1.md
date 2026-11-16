# Proposed secondary protocol v1 — approval required

Created before secondary training. Status: proposed, not frozen or approved.

Primary mechanistic comparison: A5 versus A4 isolates phase sharing for identically distributed smooth affine fields. Primary adaptive comparison: A6 versus spatially shifted-gate A7 tests image alignment of reliability. A6 versus A2 is a practical comparator, not a single-factor causal contrast. A6's clipped gain and residual formulation differ from A5 and require magnitude audits.

Primary endpoint: patient-mean full-volume out-of-fold Dice across the same five patient splits, one prespecified seed 20261031. All folds must have complete case coverage. Fold0 is exploratory because its historical results motivated the method. A claim of independent confirmation needs a separately approved source-heldout protocol or additional untouched development evaluation; do not present all reused folds as untouched hypothesis confirmation.

Secondary endpoints: Dice lower decile, severe failure Dice<0.1, HD95 mm with undefined/infinite values retained, surface Dice2mm, absolute/relative tumor volume error and source-stratified effects. Tumor-size strata: expert volume <1, 1–10, >10 cm3, specified now; do not tune bins to significance. Report counts and empty-mask policies from the existing metric implementation.

Statistical plan: patient-paired bootstrap 10000 replicates seeded20261031; paired permutation tests with two-sided alternatives; Holm correction for prespecified A5–A4, A6–A7 and A6–A2 contrasts. Report absolute effect sizes and confidence intervals. Source strata are exploratory and sample-size limited. No selective reporting of favorable folds or endpoints.

Exploratory advancement criterion: engineering and realism gates pass, no unresolved catastrophic failure, and effect estimates justify compute. A nominal Dice increase alone is insufficient. Proposed meaningful improvement target: +0.01 mean Dice, with confidence interval excluding0 in the final paired study, and severe-failure risk not worsened by >1 percentage point; these are research thresholds, not validated clinical significance. Changes require a dated amendment before new outcomes are viewed.

A6 hypothesis is falsified for this protocol if image-aligned reliability gives no reproducible advantage over matched shifted/constant controls, or produces implausible artifacts/harms failures. If A2 wins, report that flexible phase perturbations performed best under tested conditions. Do not silently redesign and call the same evaluation confirmatory.

All candidates inherit backbone, normalization, splits, loss, optimizer, 1000-epoch schedule, inference and postprocessing from the frozen originals. No Yunnan calibration/evaluation. Explicit approval, all25 original gates, exact code/config/split hashes, completed integration/real-data tests, backup proof and current authorized GPU availability are required before launch.
