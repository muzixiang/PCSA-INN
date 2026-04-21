# PCSA-INN: Physics-informed Conditional Spectral Attention Invertible Neural Network
*Official PyTorch Implementation for Anonymous Peer Review*

This repository provides the official PyTorch implementation of the PCSA-INN architecture, a novel framework designed for robust, pixel-wise atmospheric correction of multispectral satellite imagery over ocean surfaces. By leveraging the mathematical rigor of Conditional Invertible Neural Networks (cINN) and deep multimodal fusion, PCSA-INN directly learns the bidirectional mapping between Top-of-Atmosphere (TOA) radiance and Bottom-of-Atmosphere (BOA) surface reflectance.

---

## 💡 Key Architectural Highlights

- **Exact Mathematical Invertibility**: Built upon the `FrEIA` framework, ensuring a strict, volume-preserving bijective mapping. This allows the model to learn both the forward physics simulation (R_rs -> L_TOA) and the inverse atmospheric correction (L_TOA -> R_rs) within a single, unified architecture.
- **Orthogonal Channel Mixing**: Employs Householder reflections instead of standard 1x1 convolutions for numerically stable, high-dimensional feature permutations between affine coupling blocks.
- **Multimodal Physical Fusion**: Utilizes a Feature Tokenizer Transformer (FT-Transformer) to tokenize unstructured meteorological priors (e.g., Ozone, Wind Speed) into a continuous latent manifold, which adaptively conditions the affine transformations.
- **Cross-Attention Subnet**: The core affine coupling sub-networks are implemented as Transformer Decoders, where spectral/latent variables (as *queries*) attend to the meteorological embeddings (as *memory*), enforcing pixel-level physical modulation.
- **Robust Posterior Sampling**: A **Mean-Shift** based strategy is employed on Monte Carlo samples during inference to identify the mode of the posterior distribution, yielding more physically plausible and stable solutions than simple averaging, especially in ill-posed, multi-modal scenarios.

---

## 📦 Pre-trained Model Weights

For peer review reproducibility, the pre-trained weights for the PCSA-INN framework are hosted via GitHub Releases to accommodate the file size.

- **Download Link:** [[PCSA-INN v1.0 Release Weights](https://github.com/muzixiang/PCSA-INN/releases/tag/v1.0](https://github.com/muzixiang/PCSA-INN/releases/download/v1.0/best_model.pth))

**Setup:** Please download the `best_model.pth` file from the link above and place it directly in the **root directory** of this repository before executing the inference scripts.

---


## 📂 Repository Structure

To balance Intellectual Property (IP) protection with peer-review verifiability, this codebase is strictly streamlined to demonstrate the inference pipelines. 
*(Note: Core architectures are compiled, and training scripts are omitted. See details below).*

    .
    ├── configs/
    │   └── config.py          # Centralized hyperparameter and dimensionality management
    ├── data/
    │   └── dataset.py         # Inference data loader and dynamic dummy data generator
    ├── models/                # [Compiled] Core algorithm implementations obfuscated into bytecode
    │   ├── INN.py             # Main cINN architecture definition
    │   ├── FT_Transformer.py  # Condition Encoder (Meteorological Priors)
    │   └── AffineCouplingLayer.py # Cross-Attention based affine subnet
    ├── inference_correction.py# Inverse inference script (Atmospheric Correction)
    ├── inference_forward.py   # Forward inference script (RTM Surrogate Validation)
    ├── best_model.pth         # Pre-trained model weights
    └── requirements.txt       # Dependencies

---

## 🔒 Important Note on Code Availability (Peer Review Phase)

To protect pending patent applications during the double-blind review process, specific protective measures have been applied:

1. **Bytecode Obfuscation:** The foundational mathematical topologies and point-to-point attention mechanisms within the `models/` directory have been compiled into Python bytecode.
2. **Inference-Only Provision:** Training scripts, custom physical loss functions, and offline data parsing pipelines have been deliberately excluded. 

**Purpose of this Repository:**
This package is provided **strictly for execution and structural verification**. By utilizing the provided plaintext inference scripts (`inference_forward.py` and `inference_correction.py`), reviewers can seamlessly execute the code to verify the mathematical invertibility, architectural integrity, and I/O pipeline of the PCSA-INN framework without accessing the plaintext core IP.

**Future Code Availability:**
Due to ongoing patent applications and institutional intellectual property guidelines, the core modules of this repository are currently protected. Following the conclusion of the peer-review process and the finalization of these IP protections, we intend to make the functional components of the PCSA-INN framework available to the research community, subject to institutional licensing policies.

---

## ⚙️ Installation

We recommend using `conda` to ensure dependency isolation:

    conda create -n pcsainn python=3.9
    conda activate pcsainn
    pip install -r requirements.txt

---

## 🚀 Quick Start (Simulation Mode for Peer Review)

The authentic remote sensing datasets (Sentinel-3 L1B/L2 and ERA5 netCDF files) are on the terabyte scale and fall under institutional privacy constraints. 

To allow reviewers to effortlessly test the pipeline execution, we have implemented an **Automatic Dummy Generation Mechanism**. When you run the inference scripts, the `dataset.py` module will automatically synthesize physically-dimensioned Gaussian tensors (simulated TOA radiance, BOA reflectance, and meteorological priors) to mimic real satellite pixel sequences.

### Pipeline Verification

The provided scripts will automatically load the pre-trained `best_model.pth` and process the simulated data to verify both mapping directions.

    # 1. Verify the Inverse Pipeline (Atmospheric Correction: L_TOA -> R_rs)
    # This executes the Monte Carlo sampling and Mean-Shift posterior mode seeking.
    python inference_correction.py

    # 2. Verify the Forward Pipeline (RTM Surrogate Validation: R_rs -> L_TOA)
    # This validates the forward bijective mapping and extracts the latent variable Z.
    python inference_forward.py
