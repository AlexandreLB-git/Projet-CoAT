TITLE

Algorithmic optimization and implementation of a complexity-based classifier (CoAT)

ABSTRACT

This project investigates and optimizes the CoAT classifier (Complexity-based Analogical Transfer). Starting from a naive Python implementation with O(n^3) complexity, we analyse the problem and propose an optimized algorithm that only considers triplets involving the new case, reducing complexity to O(n^2). We also deliver a vectorized PyTorch implementation leveraging GPU acceleration.

PROJECT DESCRIPTION

Theoretical analysis of the incompatibility function used by CoAT and mathematical justification for complexity reduction.

Implementations compared:
Naive Python version (triplet loops).
Optimized Python version (iterate only on relevant pairs).
Vectorized PyTorch version (parallel GPU computation).

EXPERIMENTAL APPLICATION: binary classification of points in the plane (prediction of class colours).

Experimental protocol: synthetic case-base generation, prediction experiments, and runtime measurements as a function of case-base size.

MAIN RESULTS

Theoretical complexity reduction from O(n^3) to O(n^2) for the optimized algorithm.

Empirical speedups:

Optimized Python: ~30× faster than the naive Python implementation for a case-base of 100 points (≈0.5 s vs ≈15 s per predicted point).

PyTorch vectorized version: ~3750× speedup for predicting 100 points on a 100-element base in our reported experiments.


LIMITATIONS AND FUTURE WORK

Main constraint: GPU memory saturation for very large case-bases.

Suggested improvements: dynamic block splitting for GPU memory management, learning similarity metrics, benchmarking against other classifiers (K-NN, SVM), and experiments on noisy or real-world datasets.


TECHNICAL NOTE: GPU execution

During certain GPU executions, the following error may occur: “CUDA error: out of memory.”

This error does not indicate a fault in the implementation, but reflects a hardware limitation of the GPU memory. In the “coat_final.ipynb” notebook, some intermediate tensors used to construct analogies are very large. When their size exceeds the available memory on the GPU, the execution fails.

We have chosen to leave this error visible to illustrate the spatial limitations of GPUs and raise awareness of the challenges of memory management when performing parallel calculations on large tensors.

FULL REPORT

The full project report is included in the repository: CoAT_Project_Report.pdf
