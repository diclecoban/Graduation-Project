# Advisor Follow-up Narrative

Top-k feature selection gives a compact within-dataset representation, especially for HUST, but it does not solve cross-dataset transfer: the best top-k raw-transfer rows remain negative-R2.

The shift should be described as concept shift rather than only a target-scale difference. The 2.09x lifetime ratio is useful evidence of a central scale mismatch, but the stronger point is that the same early-life features do not preserve the same feature-to-lifetime relationship across MATR and HUST.

CORAL is presented as a covariate-alignment baseline. CORAL-only remains negative-R2, and CORAL-after-source CP still does not provide reliable target-domain uncertainty. This supports the claim that aligning marginal feature distributions is insufficient.

Target-side calibration and target-domain conformal prediction are the stronger controls under concept shift. A small labeled target set materially improves point prediction, and target-domain/adapted CP restores coverage much more effectively than source-calibrated CP.
