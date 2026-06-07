This repository contains the official implementation of the paper:

**Machine Unlearning in the Era of Quantum Machine Learning: An Empirical Study**

**Accepted at the 28th International Conference on Pattern Recognition (ICPR 2026).**

We present the first comprehensive empirical study of machine unlearning (MU) in hybrid quantum-classical neural networks. Although MU has been extensively investigated in classical deep learning, its behavior in variational quantum circuits (VQCs) and quantum-augmented architectures remains largely unexplored.

In this work, we:

* Adapt a broad suite of machine unlearning methods to quantum settings, including:

  * gradient-based methods
  * distillation-based methods
  * regularization-based methods
  * certified unlearning techniques
* Propose two novel unlearning strategies specifically designed for hybrid quantum-classical models.
* Evaluate unlearning under both **subset removal** and **full-class deletion** scenarios.

Experiments are conducted on **Iris**, **MNIST**, and **Fashion-MNIST** using hybrid quantum-classical neural networks. Our results show that quantum models can support effective unlearning; however, their performance is strongly influenced by:

* circuit depth
* entanglement structure
* task complexity

Shallow VQCs exhibit high intrinsic stability and limited memorization, whereas deeper hybrid models reveal more pronounced trade-offs between utility preservation, forgetting effectiveness, and alignment with oracle retraining. Across experimental settings, methods such as **EU-k**, **LCA**, and **Certified Unlearning** provide the most consistent balance across evaluation metrics.

This repository is intended to serve as a reproducible baseline and benchmark for future research in quantum machine unlearning.
