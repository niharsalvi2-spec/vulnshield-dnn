# VulnShield-DNN — Professional Build Plan

**Status:** Development protocol — repository architecture v1.0 is already frozen. This document governs *how* implementation proceeds, not *where* files live.

**Core rule:** We do not generate large amounts of code phase-by-phase in one shot. That is exactly how subtle experimental bugs, duplicated logic, inconsistent interfaces, and small accidental changes creep into a research project. VulnShield-DNN is treated like a real research software project: frozen contracts, one controlled implementation unit at a time, validation after every unit.

---

## Table of Contents

1. [The Core Rule](#1-the-core-rule)
2. [What We Are Actually Building](#2-what-we-are-actually-building)
3. [Development Has 7 Gates](#3-development-will-have-7-gates)
4. [Phase 0 — Project Governance](#4-phase-0--project-governance)
5. [Phase 1 — Environment & Configuration](#5-phase-1--environment--configuration)
6. [Phase 2 — Data Layer](#6-phase-2--data-layer)
7. [Phase 3 — Model Layer](#7-phase-3--model-layer)
8. [Phase 4 — Clean Baseline](#8-phase-4--clean-baseline)
9. [Phase 5 — Channel Catalogue](#9-phase-5--channel-catalogue)
10. [Phase 6 — Fault Injection Engine](#10-phase-6--fault-injection-engine)
11. [Phase 7 — Single-Channel Vulnerability Measurement](#11-phase-7--single-channel-vulnerability-measurement)
12. [Phase 8 — Vulnerability Ranking](#12-phase-8--vulnerability-ranking)
13. [Phase 9 — TD3 Environment](#13-phase-9--td3-environment)
14. [Phase 10 — TD3 Implementation](#14-phase-10--td3-implementation)
15. [Phase 11 — TD3 Sanity Experiment](#15-phase-11--td3-sanity-experiment)
16. [Phase 12 — Real TD3 Discovery](#16-phase-12--real-td3-discovery)
17. [Phase 13 — Baseline Methods](#17-phase-13--baseline-methods)
18. [Phase 14 — Discovery Comparison](#18-phase-14--discovery-comparison)
19. [Phase 15 — Cross-Layer Interaction](#19-phase-15--cross-layer-interaction)
20. [Phase 16 — FGSM / PGD](#20-phase-16--fgsm--pgd)
21. [Phase 17 — Protection Controller](#21-phase-17--protection-controller)
22. [Phase 18 — Fault-Aware Fine-Tuning](#22-phase-18--fault-aware-fine-tuning)
23. [Phase 19 — Protection Matrix](#23-phase-19--protection-matrix)
24. [Phase 20 — Protected Model Evaluation](#24-phase-20--protected-model-evaluation)
25. [Phase 21 — Statistical Analysis](#25-phase-21--statistical-analysis)
26. [Phase 22 — Final Analysis](#26-phase-22--final-analysis)
27. [Code Generation Policy](#27-the-most-important-change-code-generation-policy)
28. [One Implementation Unit = One Logical Responsibility](#28-one-implementation-unit--one-logical-responsibility)
29. [Implementation IDs](#29-every-code-change-gets-an-id)
30. [Module Contracts](#30-we-need-a-module-contract-before-each-implementation)
31. [No Silent Assumptions](#31-no-silent-assumptions)
32. [Decision Classification System](#32-research-decisions-will-be-explicitly-classified)
33. [No Hallucinated Results](#33-no-hallucinated-results)
34. [Reproducibility From Day One](#34-reproducibility-will-be-built-in-from-day-one)
35. [Results as Immutable Evidence](#35-results-are-immutable-evidence)
36. [Testing Strategy](#36-testing-strategy)
37. [Computational Strategy (RTX 3050)](#37-computational-strategy-for-your-rtx-3050)
38. [Full Build Sequence](#38-recommended-actual-build-sequence)
39. [Implementation Unit Response Template](#39-what-i-will-give-you-for-every-step)
40. [Change Budget](#40-we-also-need-a-change-budget)
41. [No Blind Overwriting](#41-no-blind-overwriting)
42. [Git / Version Control Strategy](#42-gitversion-control-strategy)
43. [The Three Correctness Dimensions](#43-the-most-important-scientific-principle)
44. [Definition of Done](#44-our-definition-of-done)
45. [Where We Start](#45-where-we-should-start)

---

## 1. The Core Rule

> **Specification → Contract → Minimal Implementation → Unit Test → Integration Test → Validation → Freeze → Next Component**

Not:

> "Generate all Python files → run it → fix whatever breaks."

Every implementation step in this project follows the left-hand sequence. The right-hand shortcut is explicitly disallowed.

---

## 2. What We Are Actually Building

```text
                    CIFAR-10
                       │
                       ▼
              ┌─────────────────┐
              │   Base DNN       │
              │ ResNet-18/VGG16  │
              └────────┬────────┘
                       │
                       ▼
              Clean Baseline
                       │
                       ▼
             Channel Catalogue
                       │
                       ▼
             Fault Injection
             Stuck-at-Zero
                       │
                       ▼
          Single-Channel Measurement
                       │
                       ▼
             Vulnerability Data
                       │
              ┌────────┴─────────┐
              ▼                  ▼
        TD3 Discovery        Baselines
              │          Random / Activation
              │          Gradient / Taylor
              │               / DDPG
              └────────┬─────────┘
                       ▼
             Vulnerability Ranking
                       │
              ┌────────┴──────────┐
              ▼                   ▼
       Cross-Layer           FGSM / PGD
       Interaction           Comparison
              │                   │
              └────────┬──────────┘
                       ▼
                Protection
              1% / 3% / 5% / 10%
                       │
                       ▼
             Fault-Aware Training
                       │
                       ▼
               Protected DNN
                       │
                       ▼
                  Evaluation
                       │
                       ▼
              Statistical Analysis
                       │
                       ▼
                Final Results
```

The frozen architecture explicitly defines this separation between discovery, interaction, protection, and evaluation. Nothing in this build plan changes that separation — it only governs the order and discipline of implementation.

---

## 3. Development Will Have 7 Gates

Every major phase must pass a gate before moving forward.

| Gate | Name | Question it answers |
|---|---|---|
| A | Specification | Purpose, inputs, outputs, mathematical definition, dependencies, constraints, configuration, expected failure conditions — all defined first |
| B | Interface Contract | What exactly does the module expose, before any implementation exists? |
| C | Implementation | Only the required code for this component — nothing unrelated |
| D | Unit Tests | Does the component work correctly in isolation? |
| E | Integration Test | Does it connect correctly to the previous subsystem? |
| F | Scientific Validation | Does this implementation actually represent the experiment we claim to perform? (More important than "does Python run") |
| G | Freeze | Component is locked; future changes require deliberate reason + regression testing |

### Example — Gate B in practice

```text
FaultInjector
    ├── register(...)
    ├── inject(...)
    ├── remove(...)
    └── cleanup(...)
```

We do not start writing implementation until we know what the interface means.

---

## 4. Phase 0 — Project Governance

Before any algorithm code, establish project rules.

### Deliverables

```text
README.md
CONTRIBUTING.md
CHANGELOG.md

docs/
├── architecture/
├── methodology/
├── experiments/
└── development/
```

Document: project scope, research questions, terminology, architecture, implementation order, coding conventions, reproducibility policy, experiment policy, result-handling policy.

### Critical rule

**No research result is written manually into source code.**

```python
TD3_ACCURACY = 96.4   # ❌ NEVER
```

Results must originate from actual experiments. This rule is established explicitly in the frozen architecture document.

---

## 5. Phase 1 — Environment & Configuration

**Objective:** create a deterministic, inspectable runtime environment.

### Build

```text
configs/
├── project/
├── data/
├── models/
├── faults/
├── discovery/
├── baselines/
├── interaction/
├── adversarial/
├── protection/
└── experiments/
```

### First configurations

```text
project
paths
reproducibility
dataset
model
device
logging
```

### Phase 1 validation

Executable environment check must confirm:

```text
Python version
PyTorch version
Torchvision version
CUDA availability
GPU name
CPU
RAM
configuration loading
directory validation
random seed initialization
```

Expected result:

```text
Environment: PASS
Configuration: PASS
GPU: PASS
Reproducibility setup: PASS
Repository structure: PASS
```

**No model training happens in this phase.**

---

## 6. Phase 2 — Data Layer

CIFAR-10 only.

### Components

```text
src/vulnshield/data/
├── datasets.py
├── cifar10.py
├── transforms.py
├── loaders.py
├── splits.py
└── validation.py
```

### Responsibilities

```text
Download → Storage → Transform → Train/Test → DataLoader → Validation
```

The data layer must **not** know about TD3, fault injection, protection, or vulnerability ranking. This separation is deliberate and non-negotiable.

---

## 7. Phase 3 — Model Layer

```text
models/
├── resnet/
└── vgg/
```

Implement:

- **Model A:** CIFAR-10 ResNet-18
- **Model B:** CIFAR-10 VGG-16

Do **not** immediately start the full experimental pipeline. First validate the basic model lifecycle:

```text
instantiate → forward pass → loss → backward pass → optimizer step
→ save checkpoint → reload checkpoint → verify identical behavior
```

---

## 8. Phase 4 — Clean Baseline

A scientific anchor point. Before injecting a single fault, we need a trustworthy clean model.

For each architecture:

```text
Train → Validation → Test → Checkpoint → Clean metrics
```

Preserve at:

```text
checkpoints/base_models/resnet18/
checkpoints/base_models/vgg16/
```

These are **never** overwritten by later protected models — the repository architecture explicitly separates base checkpoints from TD3/DDPG/protected checkpoints.

---

## 9. Phase 5 — Channel Catalogue

A crucial intermediate layer providing a canonical channel representation:

```text
Channel
├── model
├── layer
├── layer_index
├── channel_index
├── channel_count
├── tensor shape
└── metadata
```

Conceptually:

```text
ResNet
│
├── Conv Layer 1
│   ├── Channel 0
│   ├── Channel 1
│   └── ...
│
├── Conv Layer 2
│   ├── Channel 0
│   └── ...
│
└── ...
```

**Every discovery method must use the same catalogue.** Otherwise the comparison becomes scientifically unfair.

---

## 10. Phase 6 — Fault Injection Engine

Only now is `fault_injection/` implemented.

**Primary fault:** channel stuck-at-zero, implemented via PyTorch forward hooks.

```text
Model → Target Resolver → Hook Manager → Channel Hook → Fault Model → Forward Pass → Faulted Output
```

### Fault injector validation is extremely strict

| Test | Requirement |
|---|---|
| Test 1 | Inject channel 0 → channel 0 becomes zero, channels 1/2/... unchanged |
| Test 2 | Remove injection → original behavior returns exactly |
| Test 3 | Multiple injections → only intended channels change |
| Test 4 | Model weights before injection == weights after injection |

This validation is essential: a faulty injector could invalidate the entire research paper while still producing perfectly executable Python.

---

## 11. Phase 7 — Single-Channel Vulnerability Measurement

Scientific measurement definition:

$$
\Delta A(c) = A_{\text{clean}} - A_{\text{fault}}(c)
$$

For every selected channel:

```text
Clean accuracy → Inject channel c → Faulted accuracy → ΔA(c) → VulnerabilityRecord
```

Record structure:

```text
channel_id
layer
channel
clean_accuracy
fault_accuracy
degradation
evaluation_count
seed
model
fault_type
```

**No TD3 yet.** This phase establishes the ground-truth measurement mechanism that TD3 will later try to discover efficiently.

---

## 12. Phase 8 — Vulnerability Ranking

```text
raw vulnerability measurements → normalization → ranking → top-K
```

This becomes the common language every discovery method is judged against.

---

## 13. Phase 9 — TD3 Environment

Only after the previous layers are stable does RL design begin.

### State

Per the architecture, state contains: network topology, search progress, budget, historical degradation observations.

### Action

TD3 produces a continuous action, deterministically mapped to `(layer, channel)` via `action_mapper.py`.

### Reward

$$
R \propto \Delta A(c)
$$

with penalties for repeated/inefficient exploration, per the finalized methodology.

---

## 14. Phase 10 — TD3 Implementation

```text
actor.py
critic.py
twin_critic.py
target_networks.py
replay_buffer.py
noise.py
action_mapper.py
agent.py
trainer.py
checkpoint.py
```

### TD3 build sequence

```text
1. Actor
2. Critic
3. Twin critics
4. Target networks
5. Replay buffer
6. Exploration noise
7. Delayed policy updates
8. TD3 agent
9. Environment integration
10. Discovery runner
```

**Not all ten in one code generation request.** Each is its own controlled implementation unit, validated and frozen independently.

---

## 15. Phase 11 — TD3 Sanity Experiment

Before running expensive CIFAR-10 discovery:

```text
Toy environment → TD3 → learn → converges / behaves sensibly
```

This distinguishes a **TD3 algorithm bug** from a **research environment problem** — a significant debugging advantage before real compute is spent.

---

## 16. Phase 12 — Real TD3 Discovery

```text
CIFAR-10 → trained DNN → channel catalogue → fault environment
→ TD3 → limited injection budget → candidate channels → vulnerability ranking
```

Record **every evaluation**, not just the final ranking — discovery efficiency is itself a research metric.

---

## 17. Phase 13 — Baseline Methods

Implement: Random, Activation, Gradient, Taylor, DDPG.

### The governing scientific rule

> **Every method receives the same experimental conditions.**

Same model, dataset, fault model, candidate catalogue, evaluation budget, seeds, evaluation metric, ranking definition. Only the discovery strategy changes. This is what makes the TD3 comparison defensible.

---

## 18. Phase 14 — Discovery Comparison

Compare TD3, Random, Activation, Gradient, Taylor, DDPG on:

```text
Top-N discovery
AUC
evaluations required
injection count
ranking quality
runtime
```

The architecture explicitly separates discovery metrics from general accuracy/robustness metrics.

---

## 19. Phase 15 — Cross-Layer Interaction

For channels A and B:

```text
Measure E(A)
Measure E(B)
Measure E(A,B)
```

Interaction score:

$$
I(A,B) = E(A,B) - [E(A) + E(B)]
$$

Classification:

```text
I > +ε   → Synergistic / Amplified
|I| ≤ ε  → Additive
I < -ε   → Masking / Sub-additive
```

The architecture defines this subsystem independently from discovery and protection.

---

## 20. Phase 16 — FGSM / PGD

Only after hardware-style fault experiments are stable. This is deliberately separate:

```text
Hardware fault sensitivity   VS   Input adversarial sensitivity
```

Implement FGSM, PGD, sensitivity analysis, comparison — kept isolated so the project does not accidentally become an adversarial-ML project.

---

## 21. Phase 17 — Protection Controller

Vulnerability ranking becomes actionable.

Budgets: **1%, 3%, 5%, 10%**

```text
Vulnerability Ranking → Budget → Top-K channels → Protection Set
```

---

## 22. Phase 18 — Fault-Aware Fine-Tuning

```text
clean_loss.py
fault_loss.py
combined_loss.py
fault_aware_trainer.py
fine_tuner.py
```

Core objective:

$$
\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{clean}} + \beta \mathcal{L}_{\text{fault}}
$$

Exact implementation/configuration is frozen before large experiments run. The separation of clean/fault/combined objectives is explicitly part of the repository architecture.

---

## 23. Phase 19 — Protection Matrix

| Model | 1% | 3% | 5% | 10% |
|---|---:|---:|---:|---:|
| ResNet-18 | ✓ | ✓ | ✓ | ✓ |
| VGG-16 | ✓ | ✓ | ✓ | ✓ |

Later, protection strategies themselves are also compared. This produces a real experimental matrix rather than one cherry-picked run.

---

## 24. Phase 20 — Protected Model Evaluation

Every protected model is evaluated against:

```text
Clean
Known faults
Unseen faults
Simultaneous faults
Bit-flips
```

Exactly the conditions defined in the project architecture — no evaluation condition is skipped or substituted.

---

## 25. Phase 21 — Statistical Analysis

For multiple seeds:

$$
\mu = \frac{1}{n}\sum_i x_i
\qquad
\sigma = \sqrt{\frac{1}{n-1}\sum_i (x_i - \mu)^2}
$$

Report as **mean ± standard deviation** for appropriate metrics.

**We do not decide beforehand that TD3 is better. The experiment decides.**

---

## 26. Phase 22 — Final Analysis

```text
analysis/
```

handles: statistical analysis, ranking analysis, budget analysis, interaction analysis, sensitivity analysis, trade-off analysis.

Then generate final `results/`, `artifacts/`, `reports/`.

---

## 27. The Most Important Change: Code Generation Policy

### ❌ We will NOT do this

> "Implement Phase 1–5 completely."

This encourages huge multi-file generation and makes errors difficult to localize.

### ✅ We will do this

```text
Implement src/vulnshield/core/types.py only.
        ↓
implementation → inspect → test → fix → freeze
        ↓
Implement src/vulnshield/core/enums.py.
        ↓
test → integration → freeze
        ↓
(continue, one unit at a time)
```

---

## 28. One Implementation Unit = One Logical Responsibility

| Unit | File | Responsibility |
|---|---|---|
| F01 | `fault_model.py` | Fault-model abstraction only |
| F02 | `fault_spec.py` | Fault specification only |
| F03 | `target.py` | Target representation only |
| F04 | `target_resolver.py` | Target resolution only |
| F05 | `channel_hook.py` | Channel hook only |
| F06 | `hook_manager.py` | Hook lifecycle only |

These are implemented and validated individually, then integrated. This gives full traceability.

---

## 29. Every Code Change Gets an ID

```text
ENV-001, ENV-002, ...
DATA-001, DATA-002, ...
MODEL-001, MODEL-002, ...
FAULT-001, FAULT-002, ...
TD3-001, TD3-002, ...
```

### Example status record

```text
FAULT-004 — Channel Hook Implementation

Specification    ✓
Implementation   ✓
Unit Tests       ✓
Integration      ✓
Scientific Test  ✓
Frozen           ✓
```

This is far safer than vaguely stating "I think the fault module is done."

---

## 30. We Need a Module Contract Before Each Implementation

```text
MODULE: channel_hook.py

Purpose:
    Apply a channel-level fault to an activation tensor.

Inputs:
    activation tensor
    target channel
    fault specification

Outputs:
    modified activation tensor

Must:
    modify only requested channel

Must not:
    modify model weights
    modify unrelated channels
    persist state unexpectedly

Dependencies:
    fault_model
    target

Tests:
    ...
```

Implementation only begins after this contract exists. This eliminates a large class of hallucinated assumptions.

---

## 31. No Silent Assumptions

If the documentation says `TD3 action → discrete channel` but does not specify *exactly* how continuous coordinates are normalized/mapped, we do not silently invent a sophisticated mapping and pretend it was specified.

```text
SPECIFIED       → IMPLEMENT
NOT SPECIFIED   → FLAG → DECIDE → DOCUMENT → IMPLEMENT
```

This matters most for the project's research novelty — undocumented invented mechanics undermine the validity of any resulting claim.

---

## 32. Research Decisions Will Be Explicitly Classified

| Class | Meaning |
|---|---|
| **A** — Directly specified | From the approved project documents |
| **B** — Engineering implementation detail | Necessary to implement the spec, not a research claim |
| **C** — Research assumption | Potentially affects experimental validity |
| **D** — Experimental variable | Must be controlled/tested |
| **E** — Unknown | Requires decision or evidence before implementation |

This classification prevents accidental invention from being mistaken for specification.

---

## 33. No Hallucinated Results

At no point will we state:

```text
TD3 achieves 94%
```

unless the experiment actually produces it. Likewise, "TD3 beats DDPG" is a **hypothesis/question**, not an implementation fact. The architecture explicitly requires results to be empirical rather than hard-coded.

---

## 34. Reproducibility Will Be Built In From Day One

Every experiment eventually carries a manifest:

```text
experiment_id
timestamp
git_commit
model
dataset
fault_model
method
seed
configuration
budget
checkpoint
results
```

So that, six months later, we can answer exactly how any given number was generated.

---

## 35. Results Are Immutable Evidence

| Folder | Meaning |
|---|---|
| `configs/` | What we intended to run |
| `results/` | What actually happened |
| `artifacts/` | What we presented |
| `logs/` | What happened during execution |

This separation is already built into the frozen architecture.

---

## 36. Testing Strategy

### Level 1 — Unit
```text
test_action_mapper
test_fault_injector
test_ranking
test_reward
...
```

### Level 2 — Integration
```text
fault injection pipeline
TD3 pipeline
protection pipeline
...
```

### Level 3 — Regression
```text
clean accuracy
fault results
ranking consistency
protection consistency
```

### Level 4 — Scientific Validation

```text
Does the hook actually represent stuck-at-zero?
Does the action mapper cover all valid channels?
Does the discovery budget really limit evaluations?
Are baseline methods receiving equal budgets?
Does protection actually use the discovery ranking?
Are known/unseen channels separated correctly?
```

The architecture treats testing as necessary for research consistency — not merely conventional software engineering.

---

## 37. Computational Strategy for Your RTX 3050

We do not immediately launch massive experiments.

```text
Stage 1   CPU/GPU smoke test
Stage 2   Tiny dataset / tiny model test
Stage 3   Small real CIFAR-10 experiment
Stage 4   Single-seed pilot
Stage 5   Full single-model experiment
Stage 6   Multi-seed experiment
Stage 7   Second architecture
Stage 8   Complete matrix
```

The hardware is capable of the project — but capability does not mean GPU hours should be spent debugging unvalidated code.

---

## 38. Recommended Actual Build Sequence

```text
00  Project Governance
01  Environment
02  Configuration System
03  Core Types / Contracts
04  Data
05  Models
06  Training Infrastructure
07  Clean Baselines
08  Channel Catalogue
09  Fault Model
10  Fault Specification
11  Fault Target Resolution
12  Hook Manager
13  Channel Injection
14  Fault Validation
15  Vulnerability Measurement
16  Ranking
17  Discovery Framework
18  RL Environment
19  TD3 Components
20  TD3 Agent
21  TD3 Discovery
22  Baselines
23  Discovery Comparison
24  Cross-Layer Interaction
25  FGSM
26  PGD
27  Adversarial Comparison
28  Protection Budget
29  Channel Selection
30  Fault-Aware Loss
31  Fine-Tuning
32  Protected Models
33  Protected Evaluation
34  Full Experiment Matrix
35  Statistical Analysis
36  Figures/Tables
37  Final Research Report
```

---

## 39. Implementation Unit Response Template

Every implementation step follows this exact structure:

```text
# Implementation Unit XXXX

## 1. Objective
## 2. Scientific Purpose
## 3. Existing Dependencies
## 4. Interface Contract
## 5. Design Decisions
## 6. Files Being Modified
## 7. Files NOT Being Modified
## 8. Implementation
## 9. Tests
## 10. Validation
## 11. Acceptance Criteria
## 12. Status
```

**Only the files explicitly listed under "Files Being Modified" are changed.** No accidental expansion beyond that list.

---

## 40. We Also Need a "Change Budget"

| Size | Scope |
|---|---|
| Small | 1 file |
| Medium | 2–3 tightly coupled files |
| Large | 4–6 files maximum |

If a change requires more than 6 files, it is split into multiple implementation units. This directly addresses the risk of multiple simultaneous code changes introducing small, unintended side effects.

---

## 41. No Blind Overwriting

Before changing an existing file:

```text
Read current implementation
       ↓
Understand dependencies
       ↓
Identify exact change
       ↓
Modify only required section
       ↓
Run tests
```

An entire file is never casually regenerated because one function needs modification.

---

## 42. Git / Version Control Strategy

```text
v0.1.0   Repository + environment
v0.2.0   Data + models
v0.3.0   Clean baseline
v0.4.0   Fault injection
v0.5.0   Vulnerability pipeline
v0.6.0   TD3 discovery
v0.7.0   Baseline comparison
v0.8.0   Interaction analysis
v0.9.0   Protection
v1.0.0   Complete validated experimental system
```

Within each milestone:

```text
feature → tests → validation → commit
```

---

## 43. The Most Important Scientific Principle

Three separate things exist in this project, and passing one does **not** imply passing the others:

| Dimension | Question |
|---|---|
| Engineering correctness | Does the code work? |
| Experimental correctness | Does the experiment execute what we intended? |
| Scientific validity | Does the experiment support the research claim? |

A project can pass the first while failing the second and third. The workflow in this document exists specifically to guard against all three failure modes, not just the first.

---

## 44. Our "Definition of Done"

A module is **NOT DONE** merely because `python script.py` doesn't crash.

It is done only when:

```text
Specification       ✓
Interface           ✓
Implementation      ✓
Unit tests          ✓
Integration         ✓
Edge cases          ✓
Reproducibility     ✓
Scientific validity ✓
Documentation       ✓
Regression safety   ✓
```

Only then is the component frozen.

---

## 45. Where We Should Start

The repository skeleton is already generated from the frozen architecture. **We do not jump directly into TD3.**

```text
Repository Skeleton
        ↓
PHASE 1 — Environment + Configuration
        ↓
Validation Gate
        ↓
PHASE 2 — Data + Models
        ↓
Validation Gate
        ↓
PHASE 3 — Clean Baseline
        ↓
...
```

### First implementation target: Phase 1 — Environment & Configuration

```text
ENV-001   Repository/environment verification
ENV-002   Central configuration loader
ENV-003   Path configuration
ENV-004   Reproducibility/seed configuration
ENV-005   Device configuration
ENV-006   Configuration validation
ENV-007   Phase-1 integration test
```

**ENV-001 is implemented first, validated, then ENV-002 begins.** This is the controlled, professional build process: small enough to audit, not so fragmented that it creates pointless micro-code.

> **Note:** the frozen architecture is a structure freeze, not a license to assume every implementation detail has already been scientifically decided. Where the architecture is silent, the decision is explicitly flagged (per the classification system in §32) instead of quietly invented.
