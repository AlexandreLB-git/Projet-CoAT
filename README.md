# Algorithmic optimization of a complexity-based classifier (CoAT)

Alexandre Lebleu, Léonard Jung · Supervisor: Chunyang Fan · Sorbonne Université, May 2025

📄 **[Full report (English)](CoAT_Project_Report.pdf)** · [Version française](CoAT_Project_Report_FR.pdf) · [Notebook](coat_final.ipynb)

## Overview

CoAT (Complexity-based Analogical Transfer, [Badra, IJCAI 2020](https://doi.org/10.24963/ijcai.2020/222)) is a case-based classifier. To label a new point, it tries each candidate label and keeps the one that minimizes the number of *similarity inversions* in the dataset: triplets where x_j is closer to x_i than x_k in input space, but y_j is further from y_i than y_k in output space.

The published algorithm counts every triplet of the case base, which is O(n³) per prediction and impractical beyond a few hundred points. This project makes it fast:

1. **Theory.** We prove that the inversions inside the existing case base do not depend on the candidate label, so the prediction only depends on triplets that contain the new point. Some of those subsets provably contribute no inversion. This reduces prediction to **O(n²)** with identical outputs.
2. **Optimized Python version** that only iterates over the relevant pairs.
3. **Vectorized PyTorch version** that replaces the loops with tensor operations and runs on GPU.

## Results

Binary classification of points in the plane (synthetic data), case base of 100 points:

| Version | Complexity | Prediction time |
|---|---|---|
| Naive Python | O(n³) | ≈ 15 s per point |
| Optimized Python | O(n²) | ≈ 0.5 s per point |
| PyTorch (GPU, RTX 3050) | O(n²), vectorized | ≈ 4 ms for 100 points |

The optimized version is about 30× faster than the naive one. The PyTorch version predicts 100 points in less time than the naive version needs for a single one. GPU time stays almost flat as the case base and the number of points grow, until GPU memory saturates (see the report, Section 6).

## Repository

| File | Content |
|---|---|
| `CoAT_Project_Report.pdf` | Full report (English): proofs, algorithms, experiments |
| `CoAT_Project_Report_FR.pdf` | Original report (French) |
| `coat_final.ipynb` | Naive, optimized and PyTorch implementations, experiments and plots |

To run it, open `coat_final.ipynb`. The PyTorch part uses a CUDA GPU if available, otherwise the CPU.

**Note on GPU memory.** Some intermediate tensors grow quickly with the size of the case base. On large inputs, the notebook can raise `CUDA error: out of memory`. This is a hardware limit, not a bug, and we left it visible on purpose to illustrate the memory constraints of the vectorized approach. Splitting the tensors into blocks, as described in the report, avoids it.

## Limitations and future work

- GPU memory saturation on very large case bases (block splitting proposed).
- Evaluation focuses on runtime; prediction quality is not yet benchmarked against other classifiers (k-NN, SVM).
- Experiments use synthetic 2D data; real or noisy datasets are a natural next step.
- Learning the similarity measures from the data.

## Updates

- **October 2026:** English translation of the report.
