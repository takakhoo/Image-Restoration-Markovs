# Markov Restoration: Image Denoising Using a Pairwise MRF

## Overview
This project explores an advanced image restoration technique based on a pairwise Markov Random Field (MRF) combined with custom optimization strategies. I developed an independent implementation to restore a clean image from a noisy observation, where noise is introduced by flipping approximately 10% of the pixel values. This work delves into both the theoretical underpinnings and practical challenges of probabilistic image restoration.

## Mathematical Framework
The model is based on the energy function:
\[
E(\mathbf{x}, \mathbf{y}) = h \sum_{i} x_i - \beta \sum_{\langle i,j \rangle} x_i x_j - \eta \sum_{i} x_i y_i,
\]
where:
- \(x_i \in \{-1, +1\}\) represents the restored pixel values,
- \(y_i \in \{-1, +1\}\) represents the noisy observation,
- \(h\) is a bias term,
- \(\beta\) controls spatial smoothness (encouraging similar neighboring pixels),
- \(\eta\) enforces fidelity to the observed data.

For each pixel, the change in energy when flipping \(x_i\) is computed as:
\[
\Delta E = 2\,x_i \left(-h + \eta\,y_i + \beta \sum_{j \in N(i)} x_j\right),
\]
with \(N(i)\) denoting the four-neighbor set of pixel \(i\).

## Implementation & Experimentation
I implemented several variants of coordinate descent to minimize the energy:
- **Baseline Coordinate Descent:** A fixed update order.
- **Randomized Update Order:** Shuffling pixel updates to avoid cyclic patterns.
- **Multiple Random Restarts:** Perturbing the initial state to escape local minima.
- **Simulated Annealing:** Accepting uphill moves with probability 
  \[
  P(\text{flip}) = \exp\left(-\frac{\Delta E}{T}\right),
  \]
  with a cooling schedule \(T = \frac{T_0}{\sqrt{n+1}}\).

Through extensive parameter tuning, I achieved near-96% restoration performance, highlighting the delicate balance between model-based regularization and optimization dynamics.

## Usage
- **Prerequisites:** Python, NumPy, Matplotlib, PIL.
- **Run the Notebook:** Open `Markov_Restoration.ipynb` in Jupyter or VSCode.
- **Customization:** Experiment with parameters \(h\), \(\beta\), \(\eta\), and the annealing schedule to explore different restoration behaviors.

---
