# Scientific Methodology: Continuous Action Space Parameterization in TD3 for Discrete Fault Discovery

## 1. Problem Formulation & Combinatorial Challenge

In Deep Neural Networks, the search space for single-channel and multi-channel fault vulnerabilities is discrete and non-uniformly distributed:
* **ResNet-18:** 20 convolutional layers, $C_l \in [64, 512]$, totaling $4{,}800$ channels.
* **VGG-16:** 13 convolutional layers, $C_l \in [64, 512]$, totaling $4{,}224$ channels.

Exhaustive search requires $4{,}800$ forward passes for single faults and $\binom{4800}{2} \approx 1.15 \times 10^7$ evaluations for pairwise faults. Under a strict discovery budget of $N_{\text{queries}} = 50$, standard discrete search (e.g., bandit algorithms or tabular Q-learning) cannot explore even $1.1\%$ of the discrete channel candidates.

---

## 2. Why Continuous TD3 Parameterization?

Rather than treating each channel index as an isolated, atomic discrete arm, VulnShield-DNN establishes a continuous 2D action representation:
$$\mathbf{a} = [a_{\text{layer}},\, a_{\text{channel}}] \in [-1, 1]^2$$

Mapped via [`ActionMapper`](file:///src/vulnshield/discovery/action_mapper.py):
$$l = \text{clamp}\left( \text{round}\left( \frac{a_{\text{layer}} + 1}{2} (L - 1) \right),\, 0,\, L-1 \right)$$
$$c = \text{clamp}\left( \text{round}\left( \frac{a_{\text{channel}} + 1}{2} (C_l - 1) \right),\, 0,\, C_l - 1 \right)$$

### Key Theoretical Advantages:
1. **Architectural Smoothness & Inductive Bias:** Neighboring channels within the same feature block share similar spatial receptive fields and kernel statistics. Continuous action parameterization enables the Actor network to output a continuous density over high-gradient feature bands.
2. **Twin Critic Regularization ($\min(Q_1, Q_2)$):** Discrete Q-learning suffers from maximization bias ($\mathbb{E}[\max Q] \ge \max \mathbb{E}[Q]$), especially under small query budgets where most actions are unvisited. TD3's twin critics suppress false-positive reward spikes.
3. **Target Policy Smoothing Noise:** By adding clipped Gaussian noise $\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma^2), -c, c)$ to target actions, TD3 forces the critic to value a region of neighboring channels rather than overfitting to a single anomalous query.

---

## 3. Strict Budget Equivalence Contract

To guarantee unbiased empirical comparison against heuristic baselines (Random, Activation $L_1$, Taylor 1st-Order Gradient, and DDPG):
$$N_{\text{queries}}^{\text{TD3}} = N_{\text{queries}}^{\text{Random}} = N_{\text{queries}}^{\text{Activation}} = N_{\text{queries}}^{\text{Taylor}} = N_{\text{queries}}^{\text{DDPG}} = 50$$

All discovery algorithms are evaluated on the exact same calibration split `eval_fault` (1,000 samples) under identical forward evaluation bounds.
