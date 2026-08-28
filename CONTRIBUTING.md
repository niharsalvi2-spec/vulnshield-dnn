# Contributing to VulnShield-DNN

Thank you for contributing to the VulnShield-DNN project. This document defines the engineering, scientific, and experimental protocols that must be followed for all contributions.

---

## 1. Core Rule: 7-Gate Development Protocol

Every code contribution must follow the 7-Gate Development Protocol:

1. **Gate A — Specification:** Define inputs, outputs, mathematical formulation, error conditions, and configuration dependencies.
2. **Gate B — Interface Contract:** Define classes, methods, type annotations, and module contracts before implementing logic.
3. **Gate C — Minimal Implementation:** Implement only the specific logic allocated to the implementation unit.
4. **Gate D — Unit Testing:** Verify isolated functionality with `pytest`.
5. **Gate E — Integration Testing:** Verify integration with upstream modules.
6. **Gate F — Scientific Validation:** Confirm that the implementation faithfully models the empirical research protocol (e.g., non-destructive hooks, strictly bounded budgets).
7. **Gate G — Freeze:** Lock the component against silent modifications.

---

## 2. Code Generation & Change Budget

* **Single Responsibility:** Each implementation unit targets one logical responsibility.
* **Change Budget Limits:**
  * Small: 1 file
  * Medium: 2–3 tightly coupled files
  * Large: 4–6 files maximum (requires explicit justification)
* **No Blind Overwriting:** Existing files must be edited with surgical precision. Never regenerate an entire module when modifying a single function.

---

## 3. Prohibited Practices

* ❌ **Never Hardcode Research Results:** Do not write predefined accuracy numbers, degradation rankings, or superiority claims into source code or test fixtures.
* ❌ **Never Cross Architectural Boundaries:**
  * TD3 modules must never modify base DNN model weights.
  * Fault injection hooks must never select channels based on arbitrary heuristics.
  * Protection modules must never execute or retrain RL discovery agents.
  * Evaluation modules must never modify experimental parameters.
* ❌ **Never Create Ad-Hoc Top-Level Directories:** All files must fit into the frozen repository architecture.

---

## 4. Coding Standards

* **Python Version:** Python 3.10+
* **Type Annotations:** Full type hinting (`typing` module) across all public APIs.
* **Docstrings:** Google-style docstrings for all modules, classes, and functions.
* **Testing:** All new features must include corresponding unit tests in `tests/unit/`.
