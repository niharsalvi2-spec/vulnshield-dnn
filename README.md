# 🛡️ VulnShield-DNN

**TD3-Based Fault-Sensitive Channel Identification, Multi-Channel Interaction Dynamics, and Budget-Aware Fault Hardening for Deep Neural Networks**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4%2B-76b900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Test Suite: 128 Passed](https://img.shields.io/badge/pytest-128%20passed%20(100%25)-brightgreen.svg?logo=pytest)](file:///tests/)
[![Architecture: ResNet18 & VGG16](https://img.shields.io/badge/Models-ResNet18%20%7C%20VGG16-orange.svg)](file:///src/vulnshield/models/)
[![Dataset: CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-purple.svg)](file:///src/vulnshield/data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Key Scientific Innovations](#2-key-scientific-innovations)
3. [System Architecture & Workflow Pipeline](#3-system-architecture--workflow-pipeline)
4. [Subsystem Deep Dives](#4-subsystem-deep-dives)
   - [4.1 Data Preparation & Stratified 4-Way Splitting](#41-data-preparation--stratified-4-way-splitting)
   - [4.2 Custom CIFAR-10 Neural Architectures](#42-custom-cifar-10-neural-architectures)
   - [4.3 Clean Baseline Training Infrastructure](#43-clean-baseline-training-infrastructure)
   - [4.4 Stuck-at-Zero Fault Injection Engine](#44-stuck-at-zero-fault-injection-engine)
   - [4.5 TD3 Reinforcement Learning Discovery Agent](#45-td3-reinforcement-learning-discovery-agent)
   - [4.6 Empirical Discovery Baselines](#46-empirical-discovery-baselines)
   - [4.7 Multi-Channel Interaction & Synergy Analysis](#47-multi-channel-interaction--synergy-analysis)
   - [4.8 Budget-Constrained Fault-Aware Protection Fine-Tuning](#48-budget-constrained-fault-aware-protection-fine-tuning)
   - [4.9 Comprehensive 6-Dimensional Evaluation Suite](#49-comprehensive-6-dimensional-evaluation-suite)
   - [4.10 Publication Reporting & Figure Generation](#410-publication-reporting--figure-generation)
   - [4.11 Master Pipeline Orchestrator](#411-master-pipeline-orchestrator)
5. [Complete Repository Directory Map](#5-complete-repository-directory-map)
6. [Installation & Setup](#6-installation--setup)
7. [Step-by-Step Execution Guide](#7-step-by-step-execution-guide)
   - [7.1 One-Command Full Pipeline](#71-one-command-full-pipeline)
   - [7.2 Individual Stage-by-Stage Commands](#72-individual-stage-by-stage-commands)
8. [Hardware Feasibility & Memory Optimization](#8-hardware-feasibility--memory-optimization)
9. [Test Suite & 7-Gate Validation Protocol](#9-test-suite--7-gate-validation-protocol)
10. [Academic Citation & Authors](#10-academic-citation--authors)

---

## 1. Executive Summary & Motivation

Deep Neural Networks (DNNs) deployed in mission-critical edge environments (e.g., autonomous driving, aerospace avionics, industrial robotics, biomedical devices) are inherently vulnerable to physical hardware faults. Environmental factors such as radiation-induced **Single-Event Upsets (SEUs)**, thermal stress, memory degradation, and voltage fluctuations can flip bits or cause permanent neuron inactivation (**stuck-at-zero faults**).

Exhaustive fault simulation across all network parameters is computationally infeasible ($>4{,}000$ channels in standard vision backbones). Conventional heuristics (e.g., raw activation magnitude or isolated layer pruning) fail to capture **cross-layer non-linear fault propagation**. Furthermore, blanket redundancy (Triple Modular Redundancy) introduces unsustainable energy, latency, and silicon area overheads.

**VulnShield-DNN** provides an automated, scientifically grounded solution:
1. **Discovers** the most catastrophic fault targets using **Twin Delayed Deep Deterministic Policy Gradient (TD3)** continuous reinforcement learning under tight query constraints.
2. **Quantifies** cross-channel non-linear fault compounding and masking dynamics ($I(A, B)$ interaction metric).
3. **Hardens** target models under strict channel budgets ($1\%, 3\%, 5\%, 10\%$) through multi-objective **Fault-Aware Fine-Tuning**, preserving $\ge 99\%$ clean baseline accuracy while dramatically elevating fault tolerance.
4. **Validates** resilience across **6 distinct evaluation dimensions**, including physical IEEE 754 float32 bit-flips and decoupled gradient adversarial comparisons (FGSM / PGD-20).

---

## 2. Key Scientific Innovations

| Innovation | Description | Formulation / Core Advantage |
|:---|:---|:---|
| **Continuous RL Discovery (TD3)** | Bypasses combinatorial explosion by mapping continuous action space to discrete $(l, c)$ channel selections. | Twin Critics $\min(Q_1, Q_2)$, target policy smoothing noise, and delayed actor updates avoid overestimation bias. |
| **Cross-Layer Interaction Metric $I(A,B)$** | Discovers non-linear fault interactions when multiple channels fail concurrently. | $I(A,B) = E(A,B) - [E(A) + E(B)]$. Classifies failures as *Synergistic* ($>+1\%$), *Masking* ($<-1\%$), or *Additive*. |
| **Budget-Constrained Hardening** | Protects only the top $1\%, 3\%, 5\%, 10\%$ most vulnerable channels. | $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{clean}} + \beta \mathcal{L}_{\text{fault}} + \mathcal{L}_{\text{drift}}$. Forces neighboring channels to dynamically compensate. |
| **Physical Bit-Flip Simulation** | Emulates memory hardware upsets on 32-bit floating-point weights without re-quantization. | Directly modifies IEEE 754 bit fields: Sign (bit 31), Exponent (bits 23–30), and Mantissa (bits 0–22). |
| **Decoupled Adversarial Boundary** | Formally separates hardware reliability from adversarial perturbation bounds. | Verifies that fault-tolerant features do not compromise or confound with input gradient sensitivity (FGSM, PGD-20). |

---

## 3. System Architecture & Workflow Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   VULNSHIELD-DNN MASTER PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      [STAGE 1: CIFAR-10 Data]                                          [STAGE 2: Model Zoo]
      - Stratified Split (40k/5k/5k/1k)                                 - CIFAR-ResNet18 (4,800 channels)
      - Augmentation & Preprocessing                                    - CIFAR-VGG16 (4,224 channels)
      - Multi-worker DataLoader Container                               - Convolutional Channel Cataloger
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                     [STAGE 3: Clean Training]
                                     - SGD + Cosine Annealing (100 ep)
                                     - Gradient Clipping & Validation
                                     - Best Checkpoint Serialization
                                                  │
                                                  ▼
                                  [STAGE 4: Fault Injection Core]
                                  - Stuck-At-Zero Forward Hooks
                                  - Zero Weight Modification Guarantee
                                  - Context-Managed Automatic Cleanup
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      [STAGE 5: TD3 Discovery]                                          [STAGE 6: Baselines]
      - 4D Observation Vector                                           - Uniform Random Sampling
      - 2D Continuous Action Mapper                                     - Mean Activation Magnitude
      - Reward: r = ΔA = A_clean - A_fault                              - Taylor 1st-Order Gradient
      - Twin Critics + Delayed Updates                                  - Layer-wise Single-Critic DDPG
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                  [STAGE 7: Interaction Dynamics]
                                  - Pairwise Simultaneous Injection
                                  - Synergy / Masking Classification
                                  - Interaction Heatmaps (Seaborn)
                                                  │
                                                  ▼
                                 [STAGE 8: Fault-Aware Hardening]
                                 - Channel Budgets: 1%, 3%, 5%, 10%
                                 - Multi-Objective Loss (α=0.5, β=0.5)
                                 - L2 Representation Drift Penalty
                                                  │
                                                  ▼
                                   [STAGE 9: 6D Evaluation Suite]
                                   1. Clean Test Accuracy Drop (<=1%)
                                   2. Known Fault Set Recovery
                                   3. Unseen Fault Generalization
                                   4. Multi-Fault Stress (2, 3, 5)
                                   5. IEEE 754 Bit-Flip Robustness
                                   6. FGSM / PGD-20 Attack Comparison
                                                  │
                                                  ▼
                                  [STAGE 10: Publication Artifacts]
                                  - LaTeX Academic Tables (\begin{table})
                                  - Trade-Off Curves & Radar Plots
                                  - Comprehensive Markdown Final Report
```

---

## 4. Subsystem Deep Dives

### 4.1 Data Preparation & Stratified 4-Way Splitting
- **Source:** [`src/vulnshield/data/`](file:///src/vulnshield/data/)
- **Configuration:** [`configs/data/cifar10.yaml`](file:///configs/data/cifar10.yaml), [`configs/data/dataset_splits.yaml`](file:///configs/data/dataset_splits.yaml)
- **Split Breakdown (60,000 total images):**
  - `Train Set` (40,000 images): Clean training and fault-aware fine-tuning.
  - `Validation Set` (5,000 images): Hyperparameter tuning and model checkpoint selection.
  - `Test Set` (5,000 images): Clean accuracy evaluation.
  - `Eval-Fault Set` (1,000 images): Dedicated, deterministic subset used exclusively for fault injection and RL discovery to prevent data snooping.
- **Transforms:** Random Crop ($32\times 32$, padding=4), Random Horizontal Flip ($p=0.5$), per-channel standardization ($\mu = [0.4914, 0.4822, 0.4465]$, $\sigma = [0.2470, 0.2435, 0.2616]$).

---

### 4.2 Custom CIFAR-10 Neural Architectures
- **Source:** [`src/vulnshield/models/`](file:///src/vulnshield/models/)
- **Models Implemented:**
  1. **`CIFARResNet18`** ([`resnet.py`](file:///src/vulnshield/models/resnet.py)): Customized for $32\times 32$ spatial dimensions ($3\times 3$ initial conv with stride 1, no initial maxpool). Contains **20 Conv2d layers** and **4,800 total output channels**.
  2. **`CIFARVGG16`** ([`vgg.py`](file:///src/vulnshield/models/vgg.py)): VGG-16 with Batch Normalization customized for CIFAR-10. Contains **13 Conv2d layers** and **4,224 total output channels**.
- **Model Factory & Registry:** Dynamic instantiation by name with strict weight loading and parameter introspection via [`model_factory.py`](file:///src/vulnshield/models/model_factory.py).

---

### 4.3 Clean Baseline Training Infrastructure
- **Source:** [`src/vulnshield/training/`](file:///src/vulnshield/training/)
- **Engine:** [`BaseTrainer`](file:///src/vulnshield/training/trainer.py) executing:
  - **Loss Function:** Standard Cross-Entropy Loss with optional label smoothing ([`losses.py`](file:///src/vulnshield/training/losses.py)).
  - **Optimizer Factory:** SGD with Nesterov momentum ($0.9$), Adam, and AdamW ([`optimizer.py`](file:///src/vulnshield/training/optimizer.py)).
  - **Learning Rate Scheduler:** Cosine Annealing and Multi-Step Decay ([`scheduler.py`](file:///src/vulnshield/training/scheduler.py)).
  - **Gradient Clipping:** Strict max-norm gradient clipping ($5.0$) to stabilize training.
  - **Model Evaluator:** Measures Top-1 accuracy, Top-5 accuracy, and cross-entropy loss ([`evaluator.py`](file:///src/vulnshield/training/evaluator.py)).

---

### 4.4 Stuck-at-Zero Fault Injection Engine
- **Source:** [`src/vulnshield/fault_injection/`](file:///src/vulnshield/fault_injection/)
- **Mechanism:** Implements low-level PyTorch forward hooks ([`channel_hook.py`](file:///src/vulnshield/fault_injection/channel_hook.py)).
- **Zero-Weight-Modification Guarantee:** Model parameters $\theta$ are **never altered**. Activations on the target channel $c$ are clamped to zero in-place during the forward pass:
  $$X_{[:, c, :, :]} = 0.0$$
- **FaultInjector Context Manager:** Ensures all registered hooks are deterministically removed upon exit, even if exceptions occur during evaluation:
  ```python
  with injector.inject([("layer2.0.conv1", 42), ("layer3.1.conv2", 10)]):
      output = model(images)  # Channels 42 and 10 zeroed simultaneously
  # Hooks automatically cleared here
  ```

---

### 4.5 TD3 Reinforcement Learning Discovery Agent
- **Source:** [`src/vulnshield/discovery/`](file:///src/vulnshield/discovery/)
- **Environment:** [`FaultDiscoveryEnv`](file:///src/vulnshield/discovery/env.py) wrapping the fault injector.
  - **Observation Space (4D):**
    $$\mathbf{s} = \left[ \frac{l}{L-1},\, \frac{c}{C_{\text{max}}-1},\, \frac{A_{\text{clean}}}{100},\, \text{clip}\left(\frac{\Delta A}{100}, -1, 1\right) \right]$$
  - **Action Space (2D continuous $\in [-1, 1]^2$):** Scaled via [`ActionMapper`](file:///src/vulnshield/discovery/action_mapper.py) to a discrete $(l, c)$ channel coordinate.
  - **Reward Function:** Empirical accuracy drop:
    $$r = \Delta A(l, c) = A_{\text{clean}} - A_{\text{fault}}(l, c)$$
- **TD3 Agent Components:**
  - **Actor MLP:** [`TD3Actor`](file:///src/vulnshield/discovery/actor.py) ($4 \to 256 \to 256 \to 2$) with LayerNorm and $\tanh$ output.
  - **Twin Critic MLPs:** [`TD3TwinCritic`](file:///src/vulnshield/discovery/critic.py) computing $Q_1(s, a)$ and $Q_2(s, a)$.
  - **Replay Buffer:** Fixed-capacity circular experience replay buffer ([`replay_buffer.py`](file:///src/vulnshield/discovery/replay_buffer.py)).
  - **Delayed Policy Updates:** Actor updated every 2 critic updates.
  - **Target Policy Smoothing:** Gaussian noise ($\sigma=0.2$, clipped to $\pm 0.5$) added to target actions.
  - **Soft Target Updates:** Polyak averaging with $\tau = 0.005$.

---

### 4.6 Empirical Discovery Baselines
- **Source:** [`src/vulnshield/baselines/`](file:///src/vulnshield/baselines/)
- Evaluated under identical query budgets ($N=50$ evaluations) for comparative benchmarking:
  1. **Uniform Random Sampling:** Uniformly samples $(l, c)$ pairs across the network ([`random_baseline.py`](file:///src/vulnshield/baselines/random_baseline.py)).
  2. **Mean Activation Magnitude ($L_1$):** Accumulates mean absolute activations across a forward calibration pass ([`activation_baseline.py`](file:///src/vulnshield/baselines/activation_baseline.py)):
     $$\text{Score}(c) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{\|X_{[:, c, :, :]}\|_1}{H \times W} \right]$$
  3. **Taylor First-Order Gradient Sensitivity:** Captures gradient $\times$ activation product via backward hooks ([`gradient_baseline.py`](file:///src/vulnshield/baselines/gradient_baseline.py)):
     $$\text{Score}(c) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{\|\nabla_{X_c} \mathcal{L} \odot X_c\|_1}{H \times W} \right]$$
  4. **Layer-wise DDPG Agent:** Standard single-critic DDPG with Ornstein-Uhlenbeck noise to isolate TD3 architectural advantages ([`ddpg_baseline.py`](file:///src/vulnshield/baselines/ddpg_baseline.py)).

---

### 4.7 Multi-Channel Interaction & Synergy Analysis
- **Source:** [`src/vulnshield/interaction/`](file:///src/vulnshield/interaction/)
- **Interaction Metric:** Quantifies non-linear error compounding when two channels $A$ and $B$ are faulted simultaneously:
  $$I(A, B) = E(A, B) - [E(A) + E(B)]$$
  where $E(\cdot) = \Delta A(\cdot)$ is the accuracy degradation.
- **Categorical Classification:**
  - **Synergistic (Compounding Failure):** $I(A, B) > +1.0\%$ (Joint fault causes worse failure than the sum of individual drops).
  - **Antagonistic (Masking Effect):** $I(A, B) < -1.0\%$ (One fault suppresses or masks the degradation of the other).
  - **Additive (Independent):** $|I(A, B)| \le 1.0\%$ (Faults act independently).
- **Visualization:** Generates symmetric $N \times N$ interaction matrices and heatmaps ([`visualization.py`](file:///src/vulnshield/interaction/visualization.py)).

---

### 4.8 Budget-Constrained Fault-Aware Protection Fine-Tuning
- **Source:** [`src/vulnshield/protection/`](file:///src/vulnshield/protection/)
- **Protection Budgets:** Evaluated across $B \in \{1\%, 3\%, 5\%, 10\%\}$ of all convolutional channels:
  - **ResNet-18 ($4{,}800$ channels):** $1\% = 48$, $3\% = 144$, $5\% = 240$, $10\% = 480$ channels.
  - **VGG-16 ($4{,}224$ channels):** $1\% = 42$, $3\% = 127$, $5\% = 211$, $10\% = 422$ channels.
- **Fault-Aware Composite Loss Objective:**
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{clean}}(f(x), y) + \beta \mathcal{L}_{\text{fault}}(f_{\text{fault}}(x), y) + \frac{\lambda_{\text{drift}}}{2} \sum_{i} \|\theta_i - \theta_{\text{clean}, i}\|^2$$
  - $\alpha = 0.5, \beta = 0.5$ balances clean classification accuracy with fault robustness.
  - $\lambda_{\text{drift}} = 10^{-4}$ penalizes parameter deviation from clean representations ([`regularizer.py`](file:///src/vulnshield/protection/regularizer.py)).
- **Trainer:** [`FaultAwareTrainer`](file:///src/vulnshield/protection/fine_tuning.py) dynamically samples channels from the protected subset during training batches.

---

### 4.9 Comprehensive 6-Dimensional Evaluation Suite
- **Source:** [`src/vulnshield/evaluation/`](file:///src/vulnshield/evaluation/)
- **Evaluation Dimensions:**
  1. **Clean Test Accuracy Preservation:** Validates that clean test accuracy drop is $\le 1.0\%$ ([`clean_accuracy.py`](file:///src/vulnshield/evaluation/clean_accuracy.py)).
  2. **Known Fault Set Recovery:** Evaluates mean accuracy under faults on discovered critical channels ([`fault_evaluator.py`](file:///src/vulnshield/evaluation/fault_evaluator.py)).
  3. **Unseen Channel Generalization:** Measures accuracy under faults on 50 randomly selected channels never seen during training ([`fault_evaluator.py`](file:///src/vulnshield/evaluation/fault_evaluator.py)).
  4. **Simultaneous Multi-Fault Stress Test:** Evaluates resilience under $k \in \{2, 3, 5\}$ simultaneous random faults ([`fault_evaluator.py`](file:///src/vulnshield/evaluation/fault_evaluator.py)).
  5. **Physical Bit-Flip Simulation:** Evaluates single bit-flips in IEEE 754 float32 representation across Sign bit (31), Exponent bit (27), and Mantissa bit (10) ([`bit_flip.py`](file:///src/vulnshield/evaluation/bit_flip.py)).
  6. **Decoupled Adversarial Comparison:** Benchmarks against white-box FGSM ($\epsilon = 8/255$) and PGD-20 ($\epsilon = 8/255, \alpha = 2/255$) adversarial attacks ([`adversarial.py`](file:///src/vulnshield/evaluation/adversarial.py)).

---

### 4.10 Publication Reporting & Figure Generation
- **Source:** [`src/vulnshield/reporting/`](file:///src/vulnshield/reporting/)
- **Academic Tables:** Automatically generates publication-ready LaTeX tables (`\begin{table}`) and GitHub-Flavored Markdown tables for discovery baselines and protection budgets ([`tables.py`](file:///src/vulnshield/reporting/tables.py)).
- **Publication Figures:** Built with headless Matplotlib `Agg` backend ([`figures.py`](file:///src/vulnshield/reporting/figures.py)):
  - Discovery comparison bar charts (`artifacts/figures/resnet18_discovery_comparison.png`).
  - Budget trade-off curves (`artifacts/figures/resnet18_budget_tradeoff.png`).
  - 6-Dimensional radar evaluation plots (`artifacts/figures/resnet18_radar_evaluation.png`).
- **Automated Report Compiler:** Consolidates all findings into structured Markdown research reports in `reports/` ([`generator.py`](file:///src/vulnshield/reporting/generator.py)).

---

### 4.11 Master Pipeline Orchestrator
- **Source:** [`src/vulnshield/pipeline/`](file:///src/vulnshield/pipeline/)
- **Orchestrator:** [`VulnShieldMasterPipeline`](file:///src/vulnshield/pipeline/master_pipeline.py) runs all 8 research stages sequentially with full state management, reproducibility seed logging, and checkpoint tracking.

---

## 5. Complete Repository Directory Map

```text
VulnShield-DNN/
├── configs/                                 # Declarative YAML configurations
│   ├── data/
│   │   ├── cifar10.yaml                     # CIFAR-10 parameters & augmentations
│   │   └── dataset_splits.yaml              # Stratified split sizes (40k/5k/5k/1k)
│   ├── models/
│   │   ├── resnet18_cifar10.yaml            # ResNet-18 architecture specs
│   │   └── vgg16_cifar10.yaml               # VGG-16 architecture specs
│   ├── fault_injection/
│   │   └── stuck_at_zero.yaml               # Fault injection budget & hook mode
│   ├── protection/
│   │   ├── budgets.yaml                     # 1%, 3%, 5%, 10% channel counts
│   │   └── fault_aware_training.yaml        # Hardening hyperparameters & loss weights
│   ├── experiments/
│   │   ├── interaction.yaml                 # Multi-channel interaction thresholds
│   │   └── evaluation.yaml                  # 6-Dimensional evaluation configuration
│   └── project/
│       ├── paths.yaml                       # Dynamic project directory paths
│       └── system.yaml                      # Hardware, seed & PyTorch device specs
│
├── src/vulnshield/                          # Core Python source library
│   ├── core/
│   │   ├── constants.py                     # Global immutable project constants
│   │   ├── exceptions.py                    # Custom hierarchy (VulnShieldError)
│   │   ├── registry.py                      # Extensible component registry
│   │   └── types.py                         # Dataclasses & Type aliases
│   ├── utils/
│   │   ├── config.py                        # YAML loader & schema validator
│   │   ├── device.py                        # Hardware abstraction (CUDA / CPU)
│   │   └── reproducibility.py               # Deterministic seed management
│   ├── data/
│   │   ├── dataset.py                       # Stratified dataset indexing
│   │   ├── loaders.py                       # Multi-worker DataLoader builders
│   │   ├── transforms.py                    # Preprocessing & augmentation pipelines
│   │   └── validation.py                    # Batch shape & range sanity checks
│   ├── models/
│   │   ├── common.py                        # Channel introspection utilities
│   │   ├── resnet.py                        # CIFAR-18 architecture implementation
│   │   ├── vgg.py                           # CIFAR-VGG16 architecture implementation
│   │   ├── model_factory.py                 # Dynamic model instantiation & loading
│   │   └── model_registry.py                # Registry entry points
│   ├── training/
│   │   ├── losses.py                        # Classification loss & Top-K accuracy
│   │   ├── optimizer.py                     # SGD / Adam / AdamW builder
│   │   ├── scheduler.py                     # Cosine & MultiStep LR builder
│   │   ├── evaluator.py                     # Clean model evaluation & metrics
│   │   └── trainer.py                       # Base training loop engine (BaseTrainer)
│   ├── fault_injection/
│   │   ├── channel_hook.py                  # Low-level forward hook implementation
│   │   └── fault_injector.py                # Context-managed multi-hook injector
│   ├── vulnerability/
│   │   └── scorer.py                        # ΔA vulnerability scoring & ranking
│   ├── discovery/
│   │   ├── action_mapper.py                 # Continuous <-> Discrete channel mapper
│   │   ├── actor.py                         # TD3 Actor MLP network
│   │   ├── critic.py                        # TD3 Twin Critic MLP networks
│   │   ├── replay_buffer.py                 # Circular experience replay buffer
│   │   ├── env.py                           # Gym-style fault discovery environment
│   │   └── td3_agent.py                     # Complete TD3 RL training loop
│   ├── baselines/
│   │   ├── random_baseline.py               # Uniform random channel selector
│   │   ├── activation_baseline.py           # Mean activation magnitude selector
│   │   ├── gradient_baseline.py             # Taylor 1st-order gradient sensitivity
│   │   └── ddpg_baseline.py                 # Layer-wise single-critic DDPG agent
│   ├── interaction/
│   │   ├── metrics.py                       # I(A,B) interaction & synergy formulas
│   │   ├── synergy.py                       # Interaction aggregator & summarizer
│   │   ├── evaluator.py                     # Combinatorial pairwise evaluator
│   │   └── visualization.py                 # Symmetric matrix & heatmap generator
│   ├── protection/
│   │   ├── budget.py                        # Budget allocator (1%, 3%, 5%, 10%)
│   │   ├── losses.py                        # FaultAwareLoss (α*clean + β*fault)
│   │   ├── regularizer.py                   # WeightDriftRegularizer (L2 penalty)
│   │   └── fine_tuning.py                   # FaultAwareTrainer hardening engine
│   ├── evaluation/
│   │   ├── metrics.py                       # ComprehensiveEvaluationReport container
│   │   ├── clean_accuracy.py                # Clean accuracy drop tolerance check
│   │   ├── fault_evaluator.py               # Known, unseen & multi-fault evaluators
│   │   ├── bit_flip.py                      # IEEE 754 physical bit-flip simulator
│   │   └── adversarial.py                   # White-box FGSM & PGD-20 evaluators
│   ├── reporting/
│   │   ├── tables.py                        # LaTeX and Markdown table generators
│   │   ├── figures.py                       # Discovery bar, tradeoff, radar plots
│   │   └── generator.py                     # Master Markdown report compiler
│   └── pipeline/
│       └── master_pipeline.py               # 8-Stage Master Research Orchestrator
│
├── scripts/                                 # Reproducible CLI execution scripts
│   ├── setup/
│   │   └── check_environment.py             # Hardware & PyTorch environment check
│   ├── data/
│   │   └── download_cifar10.py              # Verified torchvision CIFAR-10 downloader
│   ├── models/
│   │   ├── inspect_model.py                 # Channel topology inspector CLI
│   │   ├── train_resnet18.py                # Clean ResNet-18 training script
│   │   ├── train_vgg16.py                   # Clean VGG-16 training script
│   │   └── evaluate_clean_model.py          # Checkpoint clean evaluator CLI
│   ├── baselines/
│   │   └── run_baselines.py                 # Discovery baselines execution CLI
│   ├── protection/
│   │   └── train_protected_model.py         # Fault-aware model hardening CLI
│   ├── experiments/
│   │   ├── run_interaction_analysis.py      # Multi-fault interaction analysis CLI
│   │   └── run_full_evaluation.py           # 6-Dimensional evaluation CLI
│   ├── analysis/
│   │   ├── generate_final_report.py         # Academic table & report compiler CLI
│   │   └── plot_comparison_figures.py       # Publication figure plotting CLI
│   └── pipeline/
│       └── run_full_pipeline.py             # One-command full pipeline runner
│
├── tests/                                   # Full unit test suite (128 tests)
│   └── unit/
│       ├── test_config.py                   # 13 tests (Schema, Paths, Seeds)
│       ├── test_data.py                     # 11 tests (Datasets, Splits, Loaders)
│       ├── test_models.py                   # 10 tests (ResNet, VGG, Factory)
│       ├── test_training.py                 # 14 tests (Loss, Optimizer, Trainer)
│       ├── test_fault_injection.py          # 14 tests (Hooks, Injector, Scorer)
│       ├── test_discovery.py                # 15 tests (TD3, Actor, Critics, Env)
│       ├── test_baselines.py                # 16 tests (Random, Act, Grad, DDPG)
│       ├── test_interaction.py              # 7 tests (Metrics, Synergy, Heatmap)
│       ├── test_protection.py               # 9 tests (Budgets, Loss, Fine-tuning)
│       ├── test_evaluation.py               # 10 tests (6D Metrics, Bit-flips, PGD)
│       ├── test_reporting.py                # 7 tests (Tables, Figures, Reports)
│       └── test_pipeline.py                 # 2 tests (Master Pipeline)
│
├── artifacts/figures/                       # Generated publication figures (.png, .pdf)
├── artifacts/tables/                        # Generated LaTeX publication tables (.tex)
├── checkpoints/                             # Model weight checkpoints (.pt)
├── reports/                                 # Generated research reports (.md)
├── results/                                 # Empirical output JSON files
├── pyproject.toml                           # PEP 518/621 build configuration
└── pytest.ini                               # PyTest configuration settings
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.4+ (Optional; full CPU fallback supported)
- 4GB+ RAM / VRAM

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/VulnShield-DNN.git
cd VulnShield-DNN
```

### Step 2: Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Package & Dependencies
```bash
# Install VulnShield-DNN in editable developer mode
pip install -e .
```

### Step 4: Verify Environment
```bash
python scripts/setup/check_environment.py
```

---

## 7. Step-by-Step Execution Guide

### 7.1 One-Command Full Pipeline
To execute the complete 8-stage research pipeline automatically (Data Download $\to$ Training $\to$ Discovery $\to$ Baselines $\to$ Interactions $\to$ Protection $\to$ 6D Evaluation $\to$ Report Generation):

```bash
# Full end-to-end execution for ResNet-18
python scripts/pipeline/run_full_pipeline.py --model resnet18 --seed 42

# Full end-to-end execution for VGG-16
python scripts/pipeline/run_full_pipeline.py --model vgg16 --seed 42
```

---

### 7.2 Individual Stage-by-Stage Commands

#### Stage 1: Download & Validate CIFAR-10
```bash
python scripts/data/download_cifar10.py
```

#### Stage 2: Train Clean Baseline Models
```bash
# Train ResNet-18 (100 epochs, SGD + Cosine Annealing)
python scripts/models/train_resnet18.py --epochs 100 --batch-size 128 --lr 0.1

# Train VGG-16 (100 epochs, SGD + Cosine Annealing)
python scripts/models/train_vgg16.py --epochs 100 --batch-size 128 --lr 0.05

# Inspect model convolutional channel structure
python scripts/models/inspect_model.py --model resnet18
```

#### Stage 3: Evaluate Clean Baseline Checkpoints
```bash
python scripts/models/evaluate_clean_model.py \
  --model resnet18 \
  --checkpoint checkpoints/base_models/resnet18/resnet18_clean_best.pt \
  --split test
```

#### Stage 4: Run Vulnerability Discovery Baselines
```bash
python scripts/baselines/run_baselines.py \
  --model resnet18 \
  --checkpoint checkpoints/base_models/resnet18/resnet18_clean_best.pt \
  --budget 50
```

#### Stage 5: Multi-Channel Fault Interaction Analysis
```bash
python scripts/experiments/run_interaction_analysis.py \
  --model resnet18 \
  --checkpoint checkpoints/base_models/resnet18/resnet18_clean_best.pt \
  --top-k 10
```

#### Stage 6: Train Fault-Aware Protected Model
```bash
# Fine-tune under 5% channel protection budget (240 channels for ResNet-18)
python scripts/protection/train_protected_model.py \
  --model resnet18 \
  --checkpoint checkpoints/base_models/resnet18/resnet18_clean_best.pt \
  --budget 0.05 \
  --epochs 30
```

#### Stage 7: Run Comprehensive 6-Dimensional Evaluation
```bash
python scripts/experiments/run_full_evaluation.py \
  --model resnet18 \
  --checkpoint checkpoints/protected/resnet18/b_5pct/resnet18_protected_b5pct_best.pt \
  --baseline-acc 93.20
```

#### Stage 8: Generate Academic LaTeX Tables, Figures & Markdown Report
```bash
# Generate LaTeX tables and Markdown report in reports/
python scripts/analysis/generate_final_report.py --model resnet18

# Generate publication-quality PNG charts in artifacts/figures/
python scripts/analysis/plot_comparison_figures.py --model resnet18
```

---

## 8. Hardware Feasibility & Memory Optimization

VulnShield-DNN was designed to run smoothly on standard commodity hardware (e.g., an **NVIDIA RTX 3050 Laptop GPU with 4GB VRAM**):

- **Forward Hook Zero-Overhead Injection:** Clamps activations in-place without duplicating feature tensors.
- **Headless Plotting:** Matplotlib is configured with the headless `Agg` backend to avoid display server dependency.
- **Deterministic Batch Sizes:** Default batch size $128$ consumes $<1.8\text{ GB}$ VRAM during training and $<0.6\text{ GB}$ VRAM during evaluation.
- **CPU Fallback:** All modules automatically fall back to CPU execution if no CUDA device is detected.

---

## 9. Test Suite & 7-Gate Validation Protocol

The entire codebase is validated by **128 unit tests** covering every functional and mathematical requirement:

```bash
# Run complete test suite with short traceback
python -m pytest tests/ -v --tb=short
```

```text
tests/unit/test_baselines.py ................                            [ 12%]
tests/unit/test_config.py .............                                  [ 22%]
tests/unit/test_data.py ...........                                      [ 31%]
tests/unit/test_discovery.py ...............                             [ 42%]
tests/unit/test_evaluation.py ..........                                 [ 50%]
tests/unit/test_fault_injection.py ..............                        [ 61%]
tests/unit/test_interaction.py .......                                   [ 67%]
tests/unit/test_models.py ..........                                     [ 75%]
tests/unit/test_pipeline.py ..                                           [ 76%]
tests/unit/test_protection.py .........                                  [ 83%]
tests/unit/test_reporting.py .......                                     [ 89%]
tests/unit/test_training.py ..............                               [100%]

======================= 128 passed, 1 warning in 57.24s =======================
```

---

## 10. Academic Citation & Authors

**Project Title:** *VulnShield-DNN: TD3-Based Fault-Sensitive Channel Identification, Cross-Layer Interaction Analysis, and Budget-Aware Robustness Enhancement for Deep Neural Networks*  
**Affiliation:** Pimpri Chinchwad College of Engineering (PCCoE), Pune  
**Academic Year:** 2025–2026  

```bibtex
@article{vulnshield_dnn_2026,
  title={VulnShield-DNN: TD3-Based Fault-Sensitive Channel Identification, Cross-Layer Interaction Analysis, and Budget-Aware Robustness Enhancement for Deep Neural Networks},
  author={Sonawane, Ishwar and Bhagat, Adhya and Salvi, Nihar},
  institution={Pimpri Chinchwad College of Engineering (PCCoE), Pune},
  year={2026}
}
```

---

*VulnShield-DNN is open-sourced under the [MIT License](LICENSE).*
