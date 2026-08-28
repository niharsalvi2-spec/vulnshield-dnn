# VulnShield-DNN — Repository Architecture v1.0

**Status:** Structure freeze recommendation — no algorithm implementation yet.
**Scope:** Grounded in the approved DL_MASTER_DOC workflow: discovery → interaction analysis → protection → evaluation, using TD3, baseline discovery methods, cross-layer fault interaction, FGSM/PGD adversarial comparison, protection budgets, and final evaluation.

---

## Table of Contents

1. [Master Repository Tree](#1-master-repository-tree)
2. [Why This Structure Is Deliberately This Large](#2-why-this-structure-is-deliberately-this-large)
3. [The Most Important Architectural Boundary](#3-the-most-important-architectural-boundary)
4. [`fault_injection/` — The Simulation Layer](#4-src-vulnshield-fault_injection)
5. [`vulnerability/` — What We Learn From Faults](#5-src-vulnshield-vulnerability)
6. [`discovery/` — The Orchestration Layer](#6-src-vulnshield-discovery)
7. [TD3 Environment Separation](#7-td3-environment-is-separate)
8. [Why `action_mapper.py` Matters](#8-why-action_mapperpy-is-important)
9. [Baselines as First-Class Citizens](#9-baselines-are-first-class-citizens)
10. [Cross-Layer Interaction Subsystem](#10-cross-layer-interaction-gets-its-own-research-subsystem)
11. [Adversarial Analysis Stays Separate](#11-adversarial-analysis-remains-separate)
12. [Protection Is Completely Downstream](#12-protection-is-completely-downstream)
13. [Why Split `clean_loss.py` / `fault_loss.py` / `combined_loss.py`](#13-why-separate-clean_losspy-fault_losspy-combined_losspy)
14. [Evaluation Never Mixes With Training](#14-evaluation-should-never-be-mixed-with-training)
15. [Metrics Are Separated](#15-metrics-are-also-separated)
16. [`experiments/` vs `results/`](#16-experiments-vs-results)
17. [`checkpoints/` Isolation](#17-checkpoints)
18. [Protection Budgets as a Physical Matrix](#18-the-four-protection-budgets-are-physically-represented)
19. [Notebooks Are Not Source of Truth](#19-notebooks-are-deliberately-not-the-source-of-truth)
20. [Scripts as Reproducible Entry Points](#20-scripts-are-the-reproducible-entry-points)
21. [Tests Are Not Optional](#21-tests-are-not-optional)
22. [Regression Tests Protect Experimental Consistency](#22-regression-tests-protect-experimental-consistency)
23. [Centralized Configuration](#23-configuration-is-centralized)
24. [Never Hard-Code Research Results](#24-one-important-correction-dont-hard-code-research-results)
25. [Traceability Table — Spec to Repository](#25-what-is-directly-grounded-in-the-approved-project)
26. [Explicitly Out of Scope](#26-one-thing-i-intentionally-did-not-add)
27. [What Not to Do — Flat `src/`](#27-another-thing-we-should-not-do)
28. [Dependency Direction to Freeze](#28-the-dependency-direction-we-should-maintain)
29. [The Five Data Zones](#29-the-repository-has-five-data-zones)
30. [Full Scientific Pipeline Map](#30-the-complete-scientific-pipeline-mapped-to-folders)
31. [What Freezing This Gives Us](#31-what-this-gives-us-during-implementation)
32. [Rule: No New Top-Level Folders](#32-one-rule-i-strongly-recommend-freezing)
33. [What to Generate Immediately](#33-what-should-be-generated-immediately)
34. [Empty Files vs `.gitkeep`](#34-important-empty-files-vs-gitkeep)
35. [Implementation Phase Order](#35-the-implementation-order-should-not-follow-the-tree-alphabetically)
36. [Final Status](#36-final-status-of-this-structure)

---

## 1. Master Repository Tree

```text
VulnShield-DNN/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .gitattributes
├── .editorconfig
├── .pre-commit-config.yaml
├── Makefile
├── pytest.ini
│
├── configs/
│   ├── README.md
│   │
│   ├── project/
│   │   ├── project.yaml
│   │   ├── paths.yaml
│   │   └── reproducibility.yaml
│   │
│   ├── data/
│   │   ├── cifar10.yaml
│   │   └── dataset_splits.yaml
│   │
│   ├── models/
│   │   ├── resnet18_cifar10.yaml
│   │   └── vgg16_cifar10.yaml
│   │
│   ├── faults/
│   │   ├── stuck_at_zero.yaml
│   │   ├── fault_injection.yaml
│   │   └── fault_evaluation.yaml
│   │
│   ├── discovery/
│   │   ├── discovery.yaml
│   │   ├── search_budgets.yaml
│   │   ├── td3.yaml
│   │   └── action_mapping.yaml
│   │
│   ├── baselines/
│   │   ├── random.yaml
│   │   ├── activation.yaml
│   │   ├── gradient.yaml
│   │   ├── taylor.yaml
│   │   └── ddpg.yaml
│   │
│   ├── interaction/
│   │   ├── cross_layer.yaml
│   │   └── interaction_metrics.yaml
│   │
│   ├── adversarial/
│   │   ├── fgsm.yaml
│   │   └── pgd.yaml
│   │
│   ├── protection/
│   │   ├── protection.yaml
│   │   ├── budgets.yaml
│   │   ├── fine_tuning.yaml
│   │   └── loss.yaml
│   │
│   └── experiments/
│       ├── development.yaml
│       ├── pilot.yaml
│       ├── full_resnet18.yaml
│       ├── full_vgg16.yaml
│       └── reproducibility.yaml
│
├── src/
│   └── vulnshield/
│       ├── __init__.py
│       ├── version.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── types.py
│       │   ├── enums.py
│       │   ├── constants.py
│       │   ├── exceptions.py
│       │   └── registry.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── cifar10.py
│       │   ├── transforms.py
│       │   ├── loaders.py
│       │   ├── splits.py
│       │   └── validation.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── model_factory.py
│       │   ├── model_registry.py
│       │   ├── common.py
│       │   │
│       │   ├── resnet/
│       │   │   ├── __init__.py
│       │   │   ├── resnet18.py
│       │   │   └── cifar_resnet18.py
│       │   │
│       │   └── vgg/
│       │       ├── __init__.py
│       │       ├── vgg16.py
│       │       └── cifar_vgg16.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── checkpointing.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   ├── losses.py
│       │   └── seed.py
│       │
│       ├── fault_injection/
│       │   ├── __init__.py
│       │   ├── injector.py
│       │   ├── hook_manager.py
│       │   ├── channel_hook.py
│       │   ├── fault_model.py
│       │   ├── fault_spec.py
│       │   ├── target.py
│       │   ├── target_resolver.py
│       │   ├── injection_context.py
│       │   ├── injection_runner.py
│       │   └── validation.py
│       │
│       ├── vulnerability/
│       │   ├── __init__.py
│       │   ├── channel.py
│       │   ├── channel_catalog.py
│       │   ├── channel_features.py
│       │   ├── channel_embeddings.py
│       │   ├── vulnerability_record.py
│       │   ├── vulnerability_score.py
│       │   ├── ranking.py
│       │   ├── ranking_utils.py
│       │   └── normalization.py
│       │
│       ├── discovery/
│       │   ├── __init__.py
│       │   │
│       │   ├── search_engine.py
│       │   ├── discovery_runner.py
│       │   ├── discovery_budget.py
│       │   ├── candidate_pool.py
│       │   ├── search_history.py
│       │   └── discovery_metrics.py
│       │
│       │   ├── td3/
│       │   │   ├── __init__.py
│       │   │   ├── agent.py
│       │   │   ├── actor.py
│       │   │   ├── critic.py
│       │   │   ├── twin_critic.py
│       │   │   ├── target_networks.py
│       │   │   ├── replay_buffer.py
│       │   │   ├── noise.py
│       │   │   ├── action_mapper.py
│       │   │   ├── trainer.py
│       │   │   └── checkpoint.py
│       │   │
│       │   └── environment/
│       │       ├── __init__.py
│       │       ├── env.py
│       │       ├── state.py
│       │       ├── action.py
│       │       ├── reward.py
│       │       ├── transition.py
│       │       ├── termination.py
│       │       └── observation.py
│       │
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── runner.py
│       │   │
│       │   ├── random/
│       │   │   ├── __init__.py
│       │   │   └── selector.py
│       │   │
│       │   ├── activation/
│       │   │   ├── __init__.py
│       │   │   ├── statistics.py
│       │   │   └── selector.py
│       │   │
│       │   ├── gradient/
│       │   │   ├── __init__.py
│       │   │   ├── sensitivity.py
│       │   │   └── selector.py
│       │   │
│       │   ├── taylor/
│       │   │   ├── __init__.py
│       │   │   ├── importance.py
│       │   │   └── selector.py
│       │   │
│       │   └── ddpg/
│       │       ├── __init__.py
│       │       ├── agent.py
│       │       ├── actor.py
│       │       ├── critic.py
│       │       ├── replay_buffer.py
│       │       ├── action_mapper.py
│       │       ├── trainer.py
│       │       └── environment.py
│       │
│       ├── interaction/
│       │   ├── __init__.py
│       │   ├── pair_selector.py
│       │   ├── cross_layer_pairs.py
│       │   ├── simultaneous_injector.py
│       │   ├── individual_effect.py
│       │   ├── combined_effect.py
│       │   ├── interaction_score.py
│       │   ├── interaction_classifier.py
│       │   └── interaction_runner.py
│       │
│       ├── adversarial/
│       │   ├── __init__.py
│       │   ├── attacks.py
│       │   ├── fgsm.py
│       │   ├── pgd.py
│       │   ├── sensitivity.py
│       │   ├── channel_analysis.py
│       │   ├── normalization.py
│       │   └── comparison.py
│       │
│       ├── protection/
│       │   ├── __init__.py
│       │   ├── budget.py
│       │   ├── channel_selector.py
│       │   ├── protection_plan.py
│       │   ├── fault_scheduler.py
│       │   ├── fine_tuner.py
│       │   ├── fault_aware_trainer.py
│       │   ├── objectives.py
│       │   ├── clean_loss.py
│       │   ├── fault_loss.py
│       │   ├── combined_loss.py
│       │   ├── protected_model.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── clean.py
│       │   ├── faulted.py
│       │   ├── unseen_faults.py
│       │   ├── simultaneous_faults.py
│       │   ├── bit_flip.py
│       │   ├── discovery_metrics.py
│       │   ├── protection_metrics.py
│       │   ├── efficiency_metrics.py
│       │   ├── robustness_metrics.py
│       │   ├── accuracy_metrics.py
│       │   └── overhead_metrics.py
│       │
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── experiment.py
│       │   ├── experiment_runner.py
│       │   ├── experiment_registry.py
│       │   ├── seeds.py
│       │   ├── repetitions.py
│       │   ├── manifests.py
│       │   └── artifacts.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── statistical.py
│       │   ├── aggregation.py
│       │   ├── comparison.py
│       │   ├── ranking_analysis.py
│       │   ├── budget_analysis.py
│       │   ├── interaction_analysis.py
│       │   ├── sensitivity_analysis.py
│       │   └── tradeoff_analysis.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── device.py
│           ├── io.py
│           ├── serialization.py
│           ├── hashing.py
│           ├── timing.py
│           ├── progress.py
│           └── reproducibility.py
│
├── scripts/
│   ├── setup/
│   │   ├── check_environment.py
│   │   ├── verify_dependencies.py
│   │   └── verify_project_structure.py
│   │
│   ├── data/
│   │   ├── download_cifar10.py
│   │   ├── prepare_data.py
│   │   └── verify_data.py
│   │
│   ├── models/
│   │   ├── train_resnet18.py
│   │   ├── train_vgg16.py
│   │   ├── evaluate_clean_model.py
│   │   └── inspect_model.py
│   │
│   ├── discovery/
│   │   ├── build_channel_catalog.py
│   │   ├── run_td3_discovery.py
│   │   ├── run_random_discovery.py
│   │   ├── run_activation_discovery.py
│   │   ├── run_gradient_discovery.py
│   │   ├── run_taylor_discovery.py
│   │   ├── run_ddpg_discovery.py
│   │   └── generate_vulnerability_ranking.py
│   │
│   ├── interaction/
│   │   ├── select_cross_layer_pairs.py
│   │   ├── run_cross_layer_analysis.py
│   │   └── classify_interactions.py
│   │
│   ├── adversarial/
│   │   ├── run_fgsm.py
│   │   ├── run_pgd.py
│   │   └── compare_fault_adversarial_sensitivity.py
│   │
│   ├── protection/
│   │   ├── generate_protection_sets.py
│   │   ├── run_fault_aware_finetuning.py
│   │   ├── evaluate_protected_models.py
│   │   └── compare_protection_budgets.py
│   │
│   ├── evaluation/
│   │   ├── run_clean_evaluation.py
│   │   ├── run_known_fault_evaluation.py
│   │   ├── run_unseen_fault_evaluation.py
│   │   ├── run_simultaneous_fault_evaluation.py
│   │   ├── run_bitflip_validation.py
│   │   ├── calculate_metrics.py
│   │   └── generate_final_comparison.py
│   │
│   └── pipeline/
│       ├── run_discovery_pipeline.py
│       ├── run_interaction_pipeline.py
│       ├── run_protection_pipeline.py
│       ├── run_evaluation_pipeline.py
│       └── run_full_pipeline.py
│
├── experiments/
│   ├── README.md
│   │
│   ├── pilot/
│   │   ├── configs/
│   │   ├── manifests/
│   │   └── notes/
│   │
│   ├── resnet18/
│   │   ├── baseline/
│   │   ├── td3/
│   │   ├── ddpg/
│   │   ├── random/
│   │   ├── activation/
│   │   ├── gradient/
│   │   ├── taylor/
│   │   ├── cross_layer/
│   │   ├── adversarial/
│   │   ├── protection/
│   │   └── evaluation/
│   │
│   └── vgg16/
│       ├── baseline/
│       ├── td3/
│       ├── ddpg/
│       ├── random/
│       ├── activation/
│       ├── gradient/
│       ├── taylor/
│       ├── cross_layer/
│       ├── adversarial/
│       ├── protection/
│       └── evaluation/
│
├── data/
│   ├── README.md
│   │
│   ├── raw/
│   │   └── .gitkeep
│   │
│   ├── interim/
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   └── .gitkeep
│   │
│   └── metadata/
│       ├── channel_catalogs/
│       │   └── .gitkeep
│       ├── dataset_metadata/
│       │   └── .gitkeep
│       └── experiment_manifests/
│           └── .gitkeep
│
├── checkpoints/
│   ├── README.md
│   │
│   ├── base_models/
│   │   ├── resnet18/
│   │   │   └── .gitkeep
│   │   └── vgg16/
│   │       └── .gitkeep
│   │
│   ├── td3/
│   │   ├── resnet18/
│   │   │   └── .gitkeep
│   │   └── vgg16/
│   │       └── .gitkeep
│   │
│   ├── ddpg/
│   │   ├── resnet18/
│   │   │   └── .gitkeep
│   │   └── vgg16/
│   │       └── .gitkeep
│   │
│   └── protected/
│       ├── resnet18/
│       │   ├── budget_1pct/
│       │   ├── budget_3pct/
│       │   ├── budget_5pct/
│       │   └── budget_10pct/
│       │
│       └── vgg16/
│           ├── budget_1pct/
│           ├── budget_3pct/
│           ├── budget_5pct/
│           └── budget_10pct/
│
├── results/
│   ├── README.md
│   │
│   ├── discovery/
│   │   ├── td3/
│   │   ├── ddpg/
│   │   ├── random/
│   │   ├── activation/
│   │   ├── gradient/
│   │   └── taylor/
│   │
│   ├── vulnerability/
│   │   ├── raw_scores/
│   │   ├── rankings/
│   │   └── summaries/
│   │
│   ├── interaction/
│   │   ├── pair_results/
│   │   ├── interaction_scores/
│   │   └── summaries/
│   │
│   ├── adversarial/
│   │   ├── fgsm/
│   │   ├── pgd/
│   │   └── comparisons/
│   │
│   ├── protection/
│   │   ├── 1pct/
│   │   ├── 3pct/
│   │   ├── 5pct/
│   │   └── 10pct/
│   │
│   ├── evaluation/
│   │   ├── clean/
│   │   ├── known_faults/
│   │   ├── unseen_faults/
│   │   ├── simultaneous_faults/
│   │   └── bitflip/
│   │
│   ├── metrics/
│   │   ├── accuracy/
│   │   ├── degradation/
│   │   ├── robustness/
│   │   ├── discovery_efficiency/
│   │   ├── protection_coverage/
│   │   └── computational_overhead/
│   │
│   └── final/
│       ├── tables/
│       ├── figures/
│       ├── statistics/
│       └── summaries/
│
├── reports/
│   ├── README.md
│   │
│   ├── exploratory/
│   │   ├── notebooks/
│   │   └── findings/
│   │
│   ├── experiments/
│   │   ├── discovery/
│   │   ├── interaction/
│   │   ├── protection/
│   │   └── evaluation/
│   │
│   └── final/
│       ├── figures/
│       ├── tables/
│       └── paper/
│
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_cifar10_baseline.ipynb
│   ├── 02_resnet18_inspection.ipynb
│   ├── 03_vgg16_inspection.ipynb
│   ├── 04_fault_injection_demo.ipynb
│   ├── 05_channel_vulnerability_analysis.ipynb
│   ├── 06_td3_discovery_analysis.ipynb
│   ├── 07_baseline_comparison.ipynb
│   ├── 08_cross_layer_interaction.ipynb
│   ├── 09_fgsm_sensitivity.ipynb
│   ├── 10_pgd_sensitivity.ipynb
│   ├── 11_fault_vs_adversarial.ipynb
│   ├── 12_protection_budget_analysis.ipynb
│   ├── 13_protected_model_analysis.ipynb
│   ├── 14_robustness_clean_accuracy_tradeoff.ipynb
│   └── 15_final_results.ipynb
│
├── tests/
│   ├── __init__.py
│   │
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_channel.py
│   │   ├── test_channel_catalog.py
│   │   ├── test_fault_spec.py
│   │   ├── test_fault_target.py
│   │   ├── test_fault_injector.py
│   │   ├── test_hook_manager.py
│   │   ├── test_vulnerability_score.py
│   │   ├── test_ranking.py
│   │   ├── test_budget.py
│   │   ├── test_action_mapper.py
│   │   ├── test_td3_actor.py
│   │   ├── test_td3_critic.py
│   │   ├── test_replay_buffer.py
│   │   ├── test_td3_environment.py
│   │   ├── test_reward.py
│   │   ├── test_baselines.py
│   │   ├── test_cross_layer.py
│   │   ├── test_interaction_score.py
│   │   ├── test_fgsm.py
│   │   ├── test_pgd.py
│   │   ├── test_protection_selection.py
│   │   ├── test_combined_loss.py
│   │   └── test_metrics.py
│   │
│   ├── integration/
│   │   ├── test_fault_injection_pipeline.py
│   │   ├── test_td3_discovery_pipeline.py
│   │   ├── test_baseline_pipeline.py
│   │   ├── test_cross_layer_pipeline.py
│   │   ├── test_adversarial_pipeline.py
│   │   ├── test_protection_pipeline.py
│   │   └── test_end_to_end_pipeline.py
│   │
│   ├── regression/
│   │   ├── test_clean_accuracy.py
│   │   ├── test_fault_results.py
│   │   ├── test_ranking_consistency.py
│   │   └── test_protection_consistency.py
│   │
│   └── fixtures/
│       ├── models/
│       ├── configs/
│       ├── tensors/
│       └── expected/
│
├── docs/
│   ├── README.md
│   │
│   ├── architecture/
│   │   ├── system_architecture.md
│   │   ├── module_architecture.md
│   │   ├── data_flow.md
│   │   ├── td3_architecture.md
│   │   ├── fault_injection_architecture.md
│   │   ├── protection_architecture.md
│   │   └── cross_layer_architecture.md
│   │
│   ├── methodology/
│   │   ├── fault_model.md
│   │   ├── vulnerability_definition.md
│   │   ├── td3_formulation.md
│   │   ├── baseline_methods.md
│   │   ├── interaction_methodology.md
│   │   ├── adversarial_methodology.md
│   │   ├── protection_methodology.md
│   │   └── evaluation_methodology.md
│   │
│   ├── experiments/
│   │   ├── experimental_protocol.md
│   │   ├── reproducibility.md
│   │   ├── random_seeds.md
│   │   └── experiment_matrix.md
│   │
│   └── development/
│       ├── coding_guidelines.md
│       ├── module_contracts.md
│       └── implementation_status.md
│
├── artifacts/
│   ├── figures/
│   │   ├── architecture/
│   │   ├── discovery/
│   │   ├── vulnerability/
│   │   ├── interaction/
│   │   ├── adversarial/
│   │   ├── protection/
│   │   └── evaluation/
│   │
│   ├── tables/
│   │   ├── discovery/
│   │   ├── vulnerability/
│   │   ├── interaction/
│   │   ├── protection/
│   │   └── evaluation/
│   │
│   └── diagrams/
│       ├── system/
│       ├── td3/
│       ├── fault_injection/
│       ├── cross_layer/
│       └── protection/
│
├── logs/
│   ├── training/
│   │   ├── td3/
│   │   ├── ddpg/
│   │   └── protection/
│   │
│   ├── experiments/
│   │   ├── discovery/
│   │   ├── interaction/
│   │   └── evaluation/
│   │
│   └── system/
│       └── .gitkeep
│
├── temp/
│   └── .gitkeep
│
└── .github/
    ├── workflows/
    │   ├── tests.yml
    │   ├── lint.yml
    │   └── structure-check.yml
    │
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   └── experiment_issue.md
    │
    └── pull_request_template.md
```

---

## 2. Why This Structure Is Deliberately This Large

We deliberately reject the minimal layout:

```text
project/
├── train.py
├── model.py
├── td3.py
├── fault.py
└── results.csv
```

That pattern is exactly how ML research projects become unmaintainable — it invites improvisation once the first "just one more script" appears.

VulnShield-DNN has several logically independent research components, so the repository mirrors the scientific architecture rather than a generic ML-project template. The approved master workflow is:

```text
Trained DNN
      ↓
Clean Baseline
      ↓
Fault Model
      ↓
Fault Injection
      ↓
TD3 Discovery
      ↓
Vulnerability Ranking
      ↓
Baseline Comparison
      ↓
Cross-Layer Analysis
      ↓
Protection Budget
      ↓
Fault-Aware Fine-Tuning
      ↓
Protected DNN
      ↓
Known + Unseen Fault Evaluation
      ↓
FGSM / PGD Comparison
      ↓
Metrics
      ↓
Final Comparison
```

This sequence is explicitly defined in the master document, and the folder structure follows it directly.

---

## 3. The Most Important Architectural Boundary

```text
                    VulnShield-DNN
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      DISCOVERY        INTERACTION       PROTECTION
          │                │                │
          ▼                ▼                ▼
        TD3             Cross-Layer     Fine-Tuning
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                       EVALUATION
```

This is not just folder organization — it prevents conceptual contamination between modules.

| Module | Forbidden behavior |
|---|---|
| TD3 | Must not modify model weights |
| Fault injector | Must not decide which channels are important |
| Protection module | Must not train TD3 |
| Evaluation module | Must not change experimental conditions |

Each module owns exactly one responsibility.

---

## 4. `src/vulnshield/fault_injection/`

The physical heart of the simulation layer. The master document specifies the primary fault as a **channel stuck-at-zero fault**, implemented via a PyTorch forward hook — the selected channel's output becomes zero while the rest of the tensor is unaffected.

```text
fault_injection/
│
├── fault_model.py
├── fault_spec.py
│
├── target.py
├── target_resolver.py
│
├── hook_manager.py
├── channel_hook.py
│
├── injection_context.py
├── injector.py
└── injection_runner.py
```

Conceptual flow:

```text
Target → Target Resolver → Hook Manager → Channel Hook → Fault Model → Fault Injection
```

---

## 5. `src/vulnshield/vulnerability/`

Represents what we learn from the faults:

```text
Layer 3 / Channel 17
        ↓
Clean Accuracy = X
Fault Accuracy = Y
        ↓
Degradation = X - Y
```

The resulting record becomes a `VulnerabilityRecord`, which flows into scoring and ranking:

```text
Vulnerability Records → Scoring → Ranking
```

This is intentionally separated from TD3 because TD3 is only **one** discovery strategy. The vulnerability representation must work identically for TD3, Random, Activation, Gradient, Taylor, and DDPG — this is what keeps the comparison fair.

---

## 6. `src/vulnshield/discovery/`

The orchestration layer, independent of which search algorithm is used:

```text
discovery/
├── search_engine.py
├── discovery_runner.py
├── discovery_budget.py
├── candidate_pool.py
└── search_history.py
```

This layer answers: *"How do we perform vulnerability discovery regardless of algorithm?"* TD3 then lives beneath it as one implementation:

```text
discovery/
└── td3/
```

---

## 7. TD3 Environment Is Separate

```text
td3/
├── agent.py
├── actor.py
├── critic.py
├── twin_critic.py
├── target_networks.py
├── replay_buffer.py
├── noise.py
├── action_mapper.py
├── trainer.py
└── checkpoint.py
```

This avoids dumping the entire TD3 implementation into a single `td3.py`. The agent interacts with a cleanly separated environment:

```text
environment/
├── env.py
├── state.py
├── action.py
├── reward.py
├── transition.py
├── termination.py
└── observation.py
```

This maps directly to the RL formulation defined in the project: state representation, action space, reward function, TD3 architecture, limited search budget, and ranking generation.

---

## 8. Why `action_mapper.py` Is Important

TD3 operates natively in a continuous action space, but the target is a discrete channel location — this mismatch must not be improvised later.

```text
TD3 Actor
     ↓
continuous action
     ↓
action_mapper.py
     ↓
candidate-location representation
     ↓
valid (layer, channel)
```

The master design maps continuous action → candidate location, rather than treating channel IDs as naturally continuous. This deserves its own dedicated component.

---

## 9. Baselines Are First-Class Citizens

Baselines do **not** belong in a single `baseline.py` — they are part of the scientific comparison, not optional extras:

```text
baselines/
│
├── random/
├── activation/
├── gradient/
├── taylor/
└── ddpg/
```

The approved project specifically compares TD3 against random selection, activation-based ranking, gradient/Taylor-based ranking, and DDPG-based search. The repository structure makes it structurally impossible to accidentally omit one.

---

## 10. Cross-Layer Interaction Gets Its Own Research Subsystem

This is one of the project's primary novelty dimensions:

```text
interaction/
│
├── pair_selector.py
├── cross_layer_pairs.py
├── simultaneous_injector.py
├── individual_effect.py
├── combined_effect.py
├── interaction_score.py
├── interaction_classifier.py
└── interaction_runner.py
```

Flow:

```text
Vulnerability Ranking
        ↓
Select Channel A / Select Channel B
        ↓
Verify different layers
        ↓
Individual A / Individual B
        ↓
A + B simultaneous fault
        ↓
Compare effects
        ↓
Interaction score
        ↓
Classify behaviour
```

The project explicitly states we must **not assume amplification** — whether combined effects are additive, masking, amplified, or otherwise different is an empirical question, which is why `interaction_classifier.py` is a justified, standalone component.

---

## 11. Adversarial Analysis Remains Separate

FGSM/PGD are a **supporting** comparison, not the main fault model:

```text
adversarial/
├── fgsm.py
├── pgd.py
├── sensitivity.py
├── channel_analysis.py
├── normalization.py
└── comparison.py
```

This boundary prevents the project from drifting into an adversarial-ML project. The master document explicitly treats FGSM/PGD as supporting analysis to compare adversarial sensitivity against hardware-fault vulnerability.

---

## 12. Protection Is Completely Downstream

```text
protection/
├── budget.py
├── channel_selector.py
├── protection_plan.py
├── fault_scheduler.py
├── fine_tuner.py
├── fault_aware_trainer.py
├── objectives.py
├── clean_loss.py
├── fault_loss.py
├── combined_loss.py
├── protected_model.py
└── checkpoint.py
```

Data flow:

```text
Vulnerability Ranking
        ↓
Protection Budget
        ↓
Top-K Selection
        ↓
Protection Plan
        ↓
Fault-Aware Fine-Tuning
        ↓
Protected Model
```

Approved budgets: **1%, 3%, 5%, 10%** — the proportion of channels eligible for targeted fault-aware fine-tuning.

---

## 13. Why Separate `clean_loss.py` / `fault_loss.py` / `combined_loss.py`

The protection objective is conceptually:

```text
Clean objective + Fault objective → Combined objective
```

Splitting these lets each component be inspected independently. This matters most when fault robustness improves but clean accuracy drops — the split lets you pinpoint which objective is driving the trade-off, rather than guessing inside one monolithic loss function.

---

## 14. Evaluation Should Never Be Mixed With Training

`evaluation/` is a fully independent top-level module with separate paths for each evaluation condition:

```text
clean.py
faulted.py
unseen_faults.py
simultaneous_faults.py
bit_flip.py
```

This matches the master workflow's explicit evaluation conditions: clean evaluation, known faults, unseen channel faults, simultaneous cross-layer faults, and limited bit-flip conditions where applicable.

---

## 15. Metrics Are Also Separated

```text
accuracy
degradation
robustness
discovery_efficiency
protection_coverage
computational_overhead
```

These map directly to the project's evaluation methodology and prevent the classic failure mode of one unmaintainable `metrics.py`.

---

## 16. `experiments/` vs `results/`

This distinction matters:

| Folder | Meaning |
|---|---|
| `experiments/` | *What experiment did we run?* — manifests, configuration |
| `results/` | *What did the experiment produce?* — CSVs, JSON, rankings, metrics, logs |

Example:

```text
experiments/resnet18/td3/   → experiment manifest/config
results/discovery/td3/      → raw output data
```

This separation is what makes experiments reproducible.

---

## 17. `checkpoints/`

Kept strictly separate:

```text
base_models/
td3/
ddpg/
protected/
```

`base_models/` must remain **untouched** — the experiment requires comparing the original clean model against the protected model, and the master workflow explicitly requires the original clean model be preserved separately so direct comparisons remain possible.

---

## 18. The Four Protection Budgets Are Physically Represented

```text
protected/
├── resnet18/
│   ├── budget_1pct/
│   ├── budget_3pct/
│   ├── budget_5pct/
│   └── budget_10pct/
│
└── vgg16/
    ├── budget_1pct/
    ├── budget_3pct/
    ├── budget_5pct/
    └── budget_10pct/
```

This produces a clean experimental matrix:

```text
                 1%    3%    5%    10%
               ┌────┬────┬────┬─────┐
ResNet-18      │    │    │    │     │
VGG-16         │    │    │    │     │
               └────┴────┴────┴─────┘
```

Each budget can be compared across TD3, DDPG, Random, Activation, Gradient, and Taylor without mixing artifacts.

---

## 19. Notebooks Are Deliberately NOT the Source of Truth

Notebooks are for exploration, visualization, debugging, and analysis — **not** where core algorithms live.

- ❌ Don't implement TD3 inside `06_td3_discovery_analysis.ipynb`
- ✅ Implement TD3 inside `src/vulnshield/discovery/td3/` and import it into the notebook

This keeps every experiment runnable through scripts as well as notebooks.

---

## 20. Scripts Are the Reproducible Entry Points

Prefer:

```text
python scripts/discovery/run_td3_discovery.py
```

over "open notebook → run cells 1–37 → hope nothing breaks." The top-level pipeline entry point is:

```text
scripts/pipeline/run_full_pipeline.py
```

which orchestrates: dataset → clean model → fault model → channel catalog → TD3 discovery → baselines → ranking → cross-layer analysis → protection → fine-tuning → evaluation → final analysis.

---

## 21. Tests Are Not Optional

Fault injection is dangerous from an **experimental correctness** standpoint. If the forward hook accidentally zeros Channel 17 while intending to modify Channel 18, the entire vulnerability ranking becomes invalid — silently.

- `tests/unit/test_fault_injector.py` is research validity protection, not routine software engineering.
- `tests/unit/test_action_mapper.py` must guarantee: continuous TD3 action → valid candidate → correct layer/channel.

---

## 22. Regression Tests Protect Experimental Consistency

If clean accuracy = X today, and three weeks later someone changes the fault hook and results silently shift, regression tests catch that drift before it contaminates downstream experiments:

```text
tests/regression/
```

---

## 23. Configuration Is Centralized

Never scatter values through Python files:

```python
learning_rate = 0.001
budget = 0.05
epochs = 20
fault_type = "zero"
```

Instead, `configs/` owns all experiment configuration (`faults/`, `discovery/`, `baselines/`, `interaction/`, `adversarial/`, `protection/`, `experiments/`), and code reads from it. This makes any experiment reproducible without relying on memory of manual edits.

---

## 24. Important Correction: Don't Hard-Code Research Results

The repository must **not** contain fabricated expected results such as:

```text
td3_accuracy = 95.7
td3_beats_ddpg = True
```

These are unknown. The master document frames these as research questions and expected evaluations — not established findings. The project must determine empirically whether TD3 improves discovery efficiency and whether the resulting vulnerability ranking improves protection.

**Rule:** `configs/` contains experimental *settings*. `results/` contains actual *measurements*. Never mix the two.

---

## 25. Traceability Table — Spec to Repository

| Component | Repository representation |
|---|---|
| CIFAR-10 | `src/vulnshield/data/` |
| ResNet-18 | `src/vulnshield/models/resnet/` |
| VGG-16 | `src/vulnshield/models/vgg/` |
| Software fault injection | `src/vulnshield/fault_injection/` |
| Stuck-at-zero channel fault | `fault_model.py`, `stuck_at_zero.yaml` |
| Channel vulnerability | `src/vulnshield/vulnerability/` |
| TD3 | `src/vulnshield/discovery/td3/` |
| RL environment | `src/vulnshield/discovery/environment/` |
| Random baseline | `baselines/random/` |
| Activation baseline | `baselines/activation/` |
| Gradient/Taylor | `baselines/gradient/`, `baselines/taylor/` |
| DDPG | `baselines/ddpg/` |
| Vulnerability ranking | `vulnerability/ranking.py` |
| Cross-layer interaction | `interaction/` |
| FGSM | `adversarial/fgsm.py` |
| PGD | `adversarial/pgd.py` |
| Fault-aware fine-tuning | `protection/` |
| Protection budget | `protection/budget.py` |
| 1%, 3%, 5%, 10% | `configs/protection/budgets.yaml` |
| Known faults | `evaluation/known_faults.py` |
| Unseen faults | `evaluation/unseen_faults.py` |
| Simultaneous faults | `evaluation/simultaneous_faults.py` |
| Bit-flip validation | `evaluation/bit_flip.py` |
| Clean accuracy | evaluation metrics |
| Faulted accuracy | evaluation metrics |
| Accuracy degradation | evaluation metrics |
| Robustness improvement | evaluation metrics |
| Discovery efficiency | evaluation metrics |
| Injection count | evaluation metrics |
| Protection coverage | evaluation metrics |
| Computational overhead | evaluation metrics |

These map directly onto the documented framework and experimental methodology.

---

## 26. Explicitly Out of Scope

No folders are created for:

```text
physical_hardware/
FPGA/
ASIC/
radiation/
runtime_recovery/
hardware_redesign/
```

The master document explicitly limits the work to **software-based** fault injection and targeted robustness enhancement — excluding physical hardware fault-injection experiments, hardware redesign, and runtime recovery mechanisms. A repository shouldn't represent research that isn't actually being done.

---

## 27. What Not to Do — Flat `src/`

Avoid:

```text
src/
├── td3.py
├── fault.py
├── model.py
├── protection.py
└── utils.py
```

The project is too large for this. Instead, each top-level folder under `src/vulnshield/` represents one research responsibility: `data/`, `models/`, `fault_injection/`, `vulnerability/`, `discovery/`, `baselines/`, `interaction/`, `adversarial/`, `protection/`, `evaluation/`, `experiments/`, `analysis/`, `utils/`.

---

## 28. The Dependency Direction to Freeze

```text
                     CORE
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        DATA        MODELS      CONFIG
          │           │
          └─────┬─────┘
                ↓
        FAULT INJECTION
                │
        ┌───────┴────────┐
        ↓                ↓
  VULNERABILITY      DISCOVERY
                       │
                 ┌─────┴─────┐
                 ↓           ↓
                TD3       BASELINES
                 │           │
                 └─────┬─────┘
                       ↓
                  RANKING
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   CROSS-LAYER                 ADVERSARIAL
   INTERACTION                 ANALYSIS
          │                         │
          └────────────┬────────────┘
                       ↓
                   PROTECTION
                       │
                       ↓
                 PROTECTED DNN
                       │
                       ↓
                  EVALUATION
                       │
                       ↓
                   ANALYSIS
```

This dependency graph matters more than any individual file name.

---

## 29. The Repository Has Five Data Zones

| Zone | Folder(s) | Purpose |
|---|---|---|
| A — Source | `src/` | Actual reusable implementation |
| B — Configuration | `configs/` | What experiment to run |
| C — Execution | `scripts/` | How to run it |
| D — Evidence | `results/`, `artifacts/`, `logs/` | What actually happened |
| E — Research documentation | `docs/`, `reports/`, `notebooks/` | How we understand and communicate it |

---

## 30. The Complete Scientific Pipeline Mapped to Folders

```text
┌─────────────────────────────────────────────────────┐
│                  01. DATASET                        │
│                 data/ + data module                 │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  02. BASE MODEL                     │
│              models/ + training/                    │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  03. FAULT MODEL                     │
│              fault_injection/                       │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              04. CHANNEL CATALOG                    │
│                 vulnerability/                      │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│               05. VULNERABILITY DISCOVERY           │
│             discovery/td3/ + environment/           │
│                         │                            │
│       ┌─────────────────┼─────────────────┐          │
│       ↓                 ↓                 ↓          │
│      TD3             RANDOM            HEURISTICS    │
│                                         + DDPG       │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                06. VULNERABILITY RANKING             │
│                  vulnerability/                      │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              07. CROSS-LAYER INTERACTION             │
│                   interaction/                       │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│               08. ADVERSARIAL COMPARISON             │
│                  adversarial/                        │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 09. PROTECTION                      │
│                  protection/                         │
│             1% / 3% / 5% / 10%                      │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                10. PROTECTED MODEL                  │
│                 checkpoints/                         │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                11. EVALUATION                       │
│                 evaluation/                         │
│                                                     │
│ Clean / Known / Unseen / Cross-layer / Bit-flip     │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                12. FINAL ANALYSIS                   │
│       analysis/ + results/ + artifacts/              │
└─────────────────────────────────────────────────────┘
```

---

## 31. What This Gives Us During Implementation

Once frozen, every future task resolves to a known location without inventing new files:

| Task | Location |
|---|---|
| Implement stuck-at-zero | `src/vulnshield/fault_injection/` |
| Implement TD3 | `src/vulnshield/discovery/td3/` |
| Implement cross-layer interaction | `src/vulnshield/interaction/` |
| Implement protection | `src/vulnshield/protection/` |
| Run the ResNet-18 TD3 experiment | `scripts/discovery/run_td3_discovery.py` |
| Find results | `results/` |

---

## 32. Rule: No New Top-Level Folders During Implementation

The top level stays fixed at:

```text
VulnShield-DNN/
│
├── configs/
├── src/
├── scripts/
├── experiments/
├── data/
├── checkpoints/
├── results/
├── reports/
├── notebooks/
├── tests/
├── docs/
├── artifacts/
├── logs/
├── temp/
└── .github/
```

If something appears to be missing, the first question is always **"which existing module does this belong to?"** — not "create `misc/`, `helpers/`, `new/`, `experimental2/`, `final_final/`." That pattern is how research repositories turn into archaeological sites.

---

## 33. What Should Be Generated Immediately

Generate the entire tree and empty files first — **do not implement algorithms yet**:

```text
Repository skeleton
        ↓
All folders created
        ↓
All predetermined files created
        ↓
Import/package structure verified
        ↓
Configuration structure verified
        ↓
Tests discovered
        ↓
Git initialized
        ↓
THEN implementation begins
```

This gives a fixed contract before any code is written.

---

## 34. Important: Empty Files vs `.gitkeep`

- Actual Python modules (`file.py`) should exist immediately as real files.
- Purely artifact/data directories only need `.gitkeep`.

Example: `data/raw/.gitkeep` signals "this directory intentionally exists but currently contains no committed dataset" — cleaner than placing placeholder files there.

---

## 35. Implementation Order (Not Alphabetical)

```text
PHASE 1   Environment + configuration
PHASE 2   CIFAR-10 + models
PHASE 3   Clean training + evaluation
PHASE 4   Channel catalog
PHASE 5   Fault injection
PHASE 6   Single-channel vulnerability measurement
PHASE 7   Ranking
PHASE 8   TD3 environment
PHASE 9   TD3
PHASE 10  Baselines
PHASE 11  Cross-layer interaction
PHASE 12  FGSM / PGD comparison
PHASE 13  Protection
PHASE 14  Protected model evaluation
PHASE 15  Full experimental matrix
PHASE 16  Final analysis
```

This order mirrors the actual scientific dependencies, not the alphabetical order of the tree.

---

## 36. Final Status of This Structure

Treat this as **VulnShield-DNN Repository Architecture v1.0**. It is grounded in the current master document, including:

- the approved research scope
- the TD3 discovery stage
- channel-level vulnerability representation
- random/activation/gradient/Taylor/DDPG baselines
- cross-layer simultaneous fault analysis
- FGSM/PGD supporting analysis
- vulnerability-guided protection
- 1/3/5/10% protection budgets
- known/unseen/simultaneous fault evaluation
- bit-flip validation where applicable
- clean/faulted accuracy, robustness, discovery efficiency, protection coverage, computational overhead
- ResNet-18 and VGG-16 on CIFAR-10
- reproducibility and experimental artifact management

No experimental results or claims about TD3 "winning" are hard-coded anywhere in this structure — those are empirical questions the implementation and experiments must answer.

### 🔒 Structure Freeze Recommendation

**Do not start writing algorithm code yet.** First generate exactly this skeleton, then validate:

```text
TREE → PACKAGE IMPORTS → CONFIG CONTRACT → MODULE CONTRACT → TEST CONTRACT → IMPLEMENTATION
```

This gives a fixed engineering blueprint: when TD3, fault injection, cross-layer analysis, or protection are eventually built, they fill predetermined slots rather than re-inventing the architecture each time.
