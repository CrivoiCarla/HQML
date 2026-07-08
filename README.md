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

## Repository Structure

```
.
├── models/
│   ├── ql_unlearning_outputs_fashion_mnist/   # trained/unlearned model outputs for Fashion-MNIST
│   ├── ql_unlearning_outputs_iris/            # trained/unlearned model outputs for Iris
│   └── ql_unlearning_outputs_mnist/           # trained/unlearned model outputs for MNIST
├── scripts/
│   ├── quantum_unlearning_iris.py             # main pipeline for the Iris dataset
│   ├── quantum_unlearning_iris_2.py           # alternative/extended pipeline for Iris
│   ├── quantum_unlearning_mnist.py            # main pipeline for MNIST
│   └── quantum_unlearning_fashion_mnist.py    # main pipeline for Fashion-MNIST
├── requirements.txt
└── README.md
```

## Programming Language

The implementation is primarily written in **Python** and uses **PyTorch** as the main deep learning framework. The required libraries and software dependencies are specified in the `requirements.txt` file.

No additional programming language is required to reproduce the experiments.

## Operating Systems

The code is designed to run on standard Python-compatible operating systems, including:

* Linux
* Windows

A Linux-based environment is recommended for full reproducibility, especially when running the complete experimental pipeline from scratch.

## Hardware Requirements

The experiments reported in the paper were executed using the following hardware configuration:

* **Processor:** AMD EPYC 7551 32-Core Processor
* **RAM:** 12 GB
* **GPU:** NVIDIA RTX 3080 Ti

Although a GPU was available, the experiments do **not rely on GPU acceleration**. The full experimental pipeline was executed sequentially and can be reproduced in a CPU-based environment.


## Execution Environment

The experiments were designed to run in a standard Python environment using the dependencies listed in `requirements.txt`.

A typical execution environment includes:
* Python 3.9 or newer
* PyTorch (`torch==2.12.0`)
* torchvision (`torchvision==0.27.0`)
* NumPy (`numpy==2.4.6`)
* scikit-learn (`scikit-learn==1.9.0`)
* PennyLane (`pennylane==0.45.0`, `pennylane_lightning==0.45.0`) — for building and simulating the variational quantum circuits
* pandas (`pandas==3.0.3`)
* scipy (`scipy==1.17.1`)
* additional dependencies listed in `requirements.txt`

To ensure reproducibility, we recommend creating a clean virtual environment before installing the required packages.

## Installation

Clone the repository:

```bash
git clone https://github.com/CrivoiCarla/HQML.git
cd HQML
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, use:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

After installing the required dependencies, the experiments can be launched directly from the corresponding script in the `scripts/` folder.

Example commands for running experiments on the supported datasets:

```bash
python scripts/quantum_unlearning_iris.py
python scripts/quantum_unlearning_mnist.py
python scripts/quantum_unlearning_fashion_mnist.py
```

An alternative/extended pipeline for the Iris dataset is also provided:

```bash
python scripts/quantum_unlearning_iris_2.py
```

## Estimated Execution Time

The complete experimental pipeline requires approximately:

```text
5–6 days
```

This estimate corresponds to running all experiments sequentially from scratch, including model training, unlearning procedures, and final evaluation.

Execution time may vary depending on the hardware configuration, software environment, and whether pre-trained model weights are used.

## Pre-Trained Models / Outputs

To facilitate reproducibility and reduce computational cost, we provide the trained model outputs used for the reported results, organized by dataset under the `models/` folder:

* `models/ql_unlearning_outputs_iris/`
* `models/ql_unlearning_outputs_mnist/`
* `models/ql_unlearning_outputs_fashion_mnist/`

These pre-computed outputs allow users to inspect and reproduce the unlearning and evaluation results without retraining all models from scratch. This is particularly useful because running the full experimental pipeline sequentially requires approximately 5–6 days.

## Code Documentation

The scripts in `scripts/` include inline comments explaining the major stages of the experimental workflow, including:

* dataset loading and preprocessing
* hybrid quantum-classical model construction
* model training
* selection of samples or classes to forget
* execution of machine unlearning methods
* post-unlearning evaluation
* comparison with oracle retraining
* generation of the reported tables

These comments are intended to make the implementation easier to understand, reproduce, and extend, especially for researchers and students working on quantum machine learning and machine unlearning.

## Notes on Reproducibility

For reproducible results, we recommend:

* using the package versions specified in `requirements.txt`
* running the code in a clean virtual environment
* keeping the dataset preprocessing settings unchanged
* setting the same random seeds across experiments
* using the provided outputs in `models/` when reproducing the reported tables

Because hybrid quantum-classical models can be sensitive to initialization, circuit structure, and optimization settings, small numerical differences may occur across hardware platforms or software versions.
