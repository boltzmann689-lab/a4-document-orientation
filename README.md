## Experimental Setup

All models were trained using the same optimization and augmentation configuration:

- **Optimizer:** AdamW
- **Learning rate:** $10^{-4}$
- **Weight decay:** $10^{-2}$
- **Loss function:** CrossEntropyLoss
- **Color augmentation:** `ColorJitter(brightness=0.2, contrast=0.2)`
- **Geometric augmentation:** `RandomRotation(degrees=20)`
- **Training duration:** 5 epochs

The baseline and proposed models were evaluated on the same fixed test set to ensure a direct and fair comparison.

---

## Experimental Results

### Performance Comparison

| Model | Architecture | Input Processing | Train Acc. | Val Acc. | Test Acc. |
|---|---|---|---:|---:|---:|
| **Baseline** | ResNet-18 | Full Image ($224 \times 224$) | 90.2% | 89.5% | 89.3% |
| **Proposed** | Multi-Scale Fusion ResNet-18 | Global Image + 4 Local Patches | **93.1%** | **91.8%** | **91.5%** |

### Key Findings

#### Performance Gain

The proposed Multi-Scale Fusion method achieved a **+2.2 percentage-point improvement** on the fixed test set compared with the standard ResNet-18 baseline:

$$
91.5\% - 89.3\% = \mathbf{+2.2\ percentage\ points}
$$

#### Convergence & Stability

Both models converged stably within 5 epochs, with no clear signs of overfitting observed during training.

#### Generalization

The proposed model achieved a validation accuracy of **91.8%** and a fixed test accuracy of **91.5%**. The close agreement between the two results indicates stable generalization to unseen real-world document images.

### Comparison with Previous Dataset

On the previous arXiv-based dataset, the baseline ResNet-18 achieved **99.89%** test accuracy. On the more diverse real-world dataset used in this project, the baseline achieved **89.3%**.

This reduction is expected because the new dataset contains substantially more heterogeneous document types, layouts, fonts, and background conditions.

---

## Training Curves

The training and validation accuracy curves are provided below.

### Baseline ResNet-18

![Baseline Training Curve](baseline_training_curve.png)

### Proposed Multi-Scale Fusion ResNet-18

![Proposed Training Curve](proposed_multiscale_fusion_curve(1).png)

---

## Installation & Usage

### 1. Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
