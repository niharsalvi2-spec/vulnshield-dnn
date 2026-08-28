# VulnShield-DNN — Reproducibility Policy

**Document Version:** 1.0  
**Status:** Active  

---

## 1. Principles of Reproducibility

To ensure every empirical result reported in papers, theses, and presentations is 100% reproducible:

1. **Explicit Seed Control:** All experiments must initialize random seeds across `torch`, `torch.cuda`, `numpy`, and `random` deterministically via `vulnshield.utils.reproducibility.set_seed()`.
2. **Centralized Configuration:** No experimental parameters may be hardcoded in scripts or source files. All hyperparameters, budgets, architectures, and attack parameters must reside in `configs/`.
3. **Experiment Manifests:** Every execution must save a JSON/YAML manifest recording:
   * Experiment ID & Timestamp
   * Random seed
   * Exact configuration dictionary
   * Model architecture & checkpoint path
   * Dataset split hashes
   * Metric results (mean $\pm$ standard deviation across seeds)
4. **Immutable Evidence:** Output metrics written to `results/` are append-only / timestamp-stamped and must never be manually fabricated or edited.
