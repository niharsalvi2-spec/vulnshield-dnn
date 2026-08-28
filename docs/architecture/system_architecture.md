# VulnShield-DNN — System Architecture

**Document Version:** 1.0  
**Status:** Frozen  

---

## 1. System Overview

VulnShield-DNN implements a decoupled, multi-stage architecture designed to assess and defend Deep Neural Networks against channel-level computational faults.

```text
┌─────────────────────────────────────────────────────────────┐
│                       VulnShield-DNN                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  DISCOVERY   │       │ INTERACTION  │       │  PROTECTION  │
│  (TD3 / Base)│       │(Cross-Layer) │       │(Fine-Tuning) │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                       ┌──────────────┐
                       │  EVALUATION  │
                       │(Clean/Fault) │
                       └──────────────┘
```

---

## 2. Core Architectural Boundaries & Contracts

To maintain experimental and scientific integrity, cross-module responsibilities are strictly bounded:

| Module | Core Responsibility | Forbidden Behavior |
| :--- | :--- | :--- |
| **`fault_injection`** | Introduces transient/permanent stuck-at-zero faults via PyTorch hooks. | Must NOT select channels or modify base weights permanently. |
| **`vulnerability`** | Maintains canonical channel catalog and converts degradation data into scores and rankings. | Must NOT implement search algorithms. |
| **`discovery`** | Orchestrates exploration (TD3 agent and baselines) under a strict fault-injection budget. | Must NOT retrain or fine-tune the base DNN. |
| **`interaction`** | Evaluates multi-channel simultaneous fault effects across network layers. | Must NOT interfere with single-channel rankings. |
| **`adversarial`** | Performs FGSM and PGD sensitivity evaluations as an independent comparative baseline. | Must NOT alter hardware fault injection pipelines. |
| **`protection`** | Selects Top-K channels according to 1%, 3%, 5%, 10% budgets and executes fault-aware fine-tuning. | Must NOT execute discovery RL policies. |
| **`evaluation`** | Benchmarks protected vs. base models across clean, known, unseen, and simultaneous fault scenarios. | Must NOT change training configurations. |

---

## 3. The Five Data Zones

1. **Zone A — Source (`src/`):** Clean, tested, modular implementation code.
2. **Zone B — Configuration (`configs/`):** Declarative, centralized YAML configurations for every experiment.
3. **Zone C — Execution (`scripts/`):** CLI entry points for automated, reproducible execution.
4. **Zone D — Evidence (`results/`, `checkpoints/`, `logs/`):** Immutable experimental records and trained model artifacts.
5. **Zone E — Research Documentation (`docs/`, `reports/`, `notebooks/`):** Scientific reports, figures, tables, and visualization notes.
