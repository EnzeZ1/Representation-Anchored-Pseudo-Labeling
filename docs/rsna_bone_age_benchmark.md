# RSNA Pediatric Bone Age benchmark

This benchmark uses only the official RSNA Pediatric Bone Age Challenge 2017 release for non-commercial research. Data remain outside Git under `/nobackup/enzez/data/rsna_bone_age/2017/` with private permissions.

Protocol `rsna-bone-age-benchmark-v1` uses Policy A: the official validation archive includes `Validation Dataset.csv` with bone-age and sex labels. Its 1,425 images are the frozen final test set. The 12,611 official training images are split once (seed 20260801) into 90% internal training and 10% internal validation, stratified by sex and 12-month bone-age bins. Split membership, identifiers, and manifests remain local and ignored.

All formal models consume pixels only. Sex is used solely for split stratification and optional post-hoc diagnostics. Images use the full-radiograph, aspect-preserving `rsna-bone-age-anatomy-preserving-v1` transform with symmetric black padding and no crop or flip.

Official sources and required attribution:

- https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017
- Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. *The RSNA Pediatric Bone Age Machine Learning Challenge*. Radiology 2018; 290(2):498–503.

The formal matrix must not launch until all six method/backbone train-and-validation-only preflights pass with zero test inference and the persistent target update count, initialization hash, and labeled exposure count match across Supervised-StepMatched, RAPL, and HPL.
