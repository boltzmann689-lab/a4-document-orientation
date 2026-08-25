# Document Orientation Classification

A deep learning pipeline for document orientation prediction ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) using a parallel dual-stream **Multi-Scale Fusion ResNet** architecture.

---

## Dataset

Instead of relying solely on standard academic papers (e.g., arXiv), the system utilizes a diverse dataset aggregated from multiple real-world document sources (>43,000 raw images):
* **SROIE (`sroie-datasetv2`)**: Scanned receipts containing various fonts, thermal paper noise, and irregular layouts.
* **CORD (`cord-1000`)**: Consolidated receipt datasets with complex background elements and distorted text alignment.
* **Receipt Dataset (`receiptdatasetssd300v2`)**: Commercial receipts with varying lighting conditions and physical folds.
* **RVL-CDIP (`the-rvlcdip-dataset-test`)**: Diverse document types including letters, forms, invoices, and reports.

### Dataset Pipeline
1. **Sampling**: 2,500 raw document images were randomly selected from the combined dataset pool.
2. **Augmentation**: Each image was rotated by $0^\circ, 90^\circ, 180^\circ,$ and $270^\circ$, generating a balanced dataset of **10,000 images**.
3. **Data Splitting**: Group-based stratified splitting ($8:1:1$) was applied to ensure all rotated variants of the same document reside within the same split:
   * **Train**: 8,000 images
   * **Validation**: 1,000 images
   * **Fixed Test**: 1,000 images

---

## Proposed Methodology

A patch-based dual-stream pipeline is introduced to extract both macro-level layout structures and micro-level textual features:

1. **Dual Inputs**: Two separate inputs are generated from a single document image: the full global image and 4 localized center patches.
2. **Global Stream**: The full page image ($224 \times 224$) passes through a ResNet-18 backbone to learn macro-level structural characteristics (page borders, whitespace margins, title positioning, layout geometry) and extracts **512 features**.
3. **Patch Stream**: The 4 center patches ($112 \times 112$ each) pass through an independent ResNet-18 backbone to capture micro-level local features (text line orientation, character alignments, diacritics/accents) and extract **2048 features** ($4 \times 512$).
4. **Feature Fusion**: Both backbones operate completely in parallel. The extracted feature maps are concatenated into a unified **2560-dimensional vector**.
5. **Classification**: The classifier receives the 2560 fused features, combining global structure with local textual cues to classify the orientation class.

### Architecture Diagram

```mermaid
flowchart TD
    A[Input Document Image] --> B["Global Image (224 x 224)"]
    A --> C["4 Patches (112 x 112)"]

    B --> D[ResNet-18 Global Backbone]
    C --> E[ResNet-18 Patch Backbone]

    D --> F[512 Features]
    E --> G[2048 Features]

    F --> H["Concatenate (2560)"]
    G --> H

    H --> I["Fully Connected (2560 -> 512 -> 4)"]
    I --> J["0° / 90° / 180° / 270°"]
