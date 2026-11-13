# Algorithmic optimization of a complexity-based classifier (CoAT)

Alexandre Lebleu, Léonard Jung · Supervisor: Chunyang Fan · Sorbonne Université, June 2025

📄 [Full report (English)](CoAT_Project_Report.pdf) · [Version française](CoAT_Project_Report_FR.pdf) · [Notebook](coat_final.ipynb)

## Overview

CoAT (Complexity-based Analogical Transfer, Badra, IJCAI 2020) is a case-based classifier. To label a new point, it tries each candidate label and keeps the one that minimizes the number of *similarity inversions* in the dataset: triplets where x_j is closer to x_i than x_k in input space, but y_j is further from y_i than y_k in output space.

The published algorithm counts every triplet of the case base, which is O(n³) per prediction and impractical beyond a few hundred points. This project makes it fast:

- **Theory.** We prove that the inversions inside the existing case base do not depend on the candidate label, so the prediction only depends on triplets that contain the new point. Some of those subsets provably contribute no inversion. This reduces prediction to O(n²) with identical outputs.
- **Optimized Python version** that only iterates over the relevant pairs.
- **Vectorized PyTorch version** that replaces the loops with tensor operations and runs on GPU.

The notebook checks numerically that the three versions give the same predictions: the O(n²) score equals the naive score minus a constant that does not depend on the label, and the PyTorch version matches the Python one on every test point.

## Results

Binary classification of points in the plane (synthetic data), case base of 100 points, run on a Google Colab GPU:

| Version | Complexity | Prediction time | Speedup |
|---|---|---|---|
| Naive Python | O(n³) | ≈ 8 s per point | |
| Optimized Python | O(n²) | ≈ 0.25 s per point | ≈ 30× vs naive |
| PyTorch (GPU) | O(n²), vectorized | ≈ 3.5 ms per point | ≈ 70× vs optimized, over 2,000× vs naive |

The O(n²) speedup comes from the algorithm and is stable across runs. The GPU speedup depends on the hardware and its load: for a single point, the measured time is mostly the fixed cost of launching GPU operations. The GPU pays off on batches: 100 points are predicted in about 10 ms with a case base of 100, and in under 20 ms for case bases up to 300 points.

## Repository

| File | Content |
|---|---|
| `CoAT_Project_Report.pdf` | Full report (English): proofs, algorithms, experiments |
| `CoAT_Project_Report_FR.pdf` | Original report (French) |
| `coat_final.ipynb` | Naive, optimized and PyTorch implementations, correctness checks, benchmarks and plots |

To run it, open `coat_final.ipynb` (for example in Google Colab with a GPU runtime). The PyTorch part uses a CUDA GPU if available, otherwise the CPU.

**Note on GPU memory.** The comparison tensors of the vectorized version grow as p × N² (p query points, N cases). On large inputs this can exhaust GPU memory. Splitting the tensors into blocks, as described in the report, avoids it.

## Limitations and future work

- GPU memory grows quickly on very large case bases (block splitting proposed).
- Evaluation focuses on runtime; prediction quality is not yet benchmarked against other classifiers (k-NN, SVM).
- Experiments use synthetic 2D data; real or noisy datasets are a natural next step.
- Learning the similarity measures from the data.

## Updates

- **November 2026:** fixed bugs in the notebook, added correctness checks across the three versions, translated the notebook to English and re-ran all benchmarks. The timings above come from this version; the report reflects the original June 2025 experiments.
- **October 2026:** English translation of the report.
