# FCHNet
# FCHNet: Frequency-guided Camouflaged Object Detection Network

## 📌 Introduction

FCHNet is a deep learning framework for **camouflaged object detection (COD)**, which leverages **frequency-domain information** and **hierarchical feature fusion** to enhance detection performance in challenging scenarios.

The model integrates:

* **Wavelet-based Frequency Reconstruction (WFR)**
* **Cross-level Frequency Aggregation (CFA)**
* **Boundary Feature Enhancement (BFE)**
* **Boundary-guided Decoder (BGD)**

to effectively capture both **spatial and frequency cues**.

---

## 🧠 Overall Architecture

FCHNet consists of the following key components:

* **Backbone**: EfficientNet-based feature extractor
* **WFR Module**: Multi-scale wavelet reconstruction for feature enhancement
* **CFA Module**: Cross-scale frequency feature aggregation
* **BFE Module**: Boundary feature enhancement
* **BGD Module**: Progressive decoding with boundary guidance

---

## 📂 Project Structure

```text
FCHNet/
├── Model/
│   ├── FCHNet.py          # Main model
│   ├── WFFMA.py           # CFA module (renamed from WFFM)
│   ├── WFR related code   # Wavelet reconstruction modules
│   ├── BFE / BGD modules  # Boundary modules
│   └── backbone files     # EfficientNet etc.
│
├── utils/
│   ├── dataloader_freq.py
│   ├── data_augmentation.py
│   ├── dct.py
│   └── metrics.py
│
├── train.py
├── inference.py
├── evaluate.py
└── README.md
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch 1.10+
* torchvision
* numpy
* opencv-python

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The model is designed for camouflaged object detection benchmarks:

* CAMO
* COD10K
* CHAMELEON
* NC4K

Prepare dataset structure like:

```text
dataset/
├── TrainDataset/
│   ├── Image/
│   ├── GT/
│   └── Edge/
```

Update dataset paths in `config.py`.

---

## 🚀 Training

```bash
python train.py
```

Key settings (in `config.py`):

* batch size
* learning rate
* dataset path
* epochs

---

## 🔍 Inference

```bash
python inference.py
```

Predictions will be saved in:

```text
prediction_maps/
```

---

## 📈 Evaluation

```bash
python evaluate.py
```

Metrics include:

* S-measure
* E-measure
* Weighted F-measure
* MAE

---

## 🧩 Key Modules

### 🔹 WFR (Wavelet Feature Reconstruction)

* Decomposes features into low/high frequency
* Reconstructs enhanced representations via IDWT

### 🔹 CFA (Cross-level Frequency Aggregation)

* Aggregates multi-scale frequency features
* Improves semantic consistency

### 🔹 BFE (Boundary Feature Enhancement)

* Extracts fine-grained edge information

### 🔹 BGD (Boundary-guided Decoder)

* Progressive decoding with boundary supervision

---

## 📌 Notes

* The project has been **cleaned and simplified**
* Unused modules and experimental files are removed
* Naming has been unified:

  * WFNet → FCHNet
  * WFFM → CFA
  * edgeconv → BFE
  * deconv → BGD

---

## 📜 License

This project is for academic research only.

---

## ✨ Acknowledgements

Thanks to prior works on:

* Camouflaged Object Detection
* Frequency Domain Learning
* Wavelet-based Feature Fusion

---

## 📬 Contact

If you have any questions, feel free to open an issue.
