# VulnShield-DNN — Coding Guidelines

**Document Version:** 1.0  
**Status:** Active  

---

## 1. Code Standards

* **Language:** Python 3.10+
* **Type Hints:** Required for all function/method signatures and public variables.
* **Style:** Follow PEP 8 style formatting.
* **Docstrings:** Google-style docstrings for all classes and functions.

Example:
```python
def calculate_degradation(clean_acc: float, faulted_acc: float) -> float:
    """Calculate the accuracy degradation caused by a simulated fault.

    Args:
        clean_acc: Top-1 clean model classification accuracy (0.0 to 1.0).
        faulted_acc: Top-1 classification accuracy under fault injection.

    Returns:
        Accuracy degradation delta (clean_acc - faulted_acc).
    """
    return clean_acc - faulted_acc
```

---

## 2. Module Implementation Protocol

1. Read existing dependencies before editing.
2. Maintain strict decoupling: depend on abstractions defined in `core/types.py` and `core/enums.py`.
3. Never use global mutable state.
4. Pass configuration objects or parameters explicitly.
5. Provide isolated unit tests in `tests/unit/` for every newly implemented class.
