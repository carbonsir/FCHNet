# Frequency-Guided Cross-Domain Learning for Lightweight Camouflaged Object Detection
> 🚧 **Note**  
> The code and pretrained models will be released after the paper is accepted for publication.

## 📌 Abstract
<p align="justify">
Camouflaged object detection (COD) faces challenges in preserving subtle structural cues due to weak boundaries and high background similarity. Existing methods often use loose fusion of frequency priors and spatial features, leading to unstable representation. This work presents a Frequency-aware Cross-domain Hierarchical Network (FCHNet) for efficient COD. The model integrates a Cross-domain Frequency Alignment module and Wavelet Feature Refinement to stabilize spatial-frequency coordination. A boundary-aware decoder further refines object contours. Experiments on CAMO, COD10K, and NC4K show FCHNet outperforms lightweight models by 3.1% in weighted F-measure with under 5M parameters, achieving efficiency and accuracy balance.
</p>
---

## 🧠 Overall Architecture

FCHNet consists of the following key components:

* **Backbone**: EfficientNet-based feature extractor
* **WFR Module**: Multi-scale wavelet feature refinement
* **CFA Module**: Cross-domain frequency alignment
* **BFE Module**: Boundary feature extraction
* **BGD Module**: Boundary-guided decoder

---

## 📂 Project Structure

```text
FCHNet/
├── Model/
│   ├── FCHNet.py          # Main model
│   ├── CFA.py             # CFA module 
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
The training and testing datasets can be downloaded from https://github.com/GewelsJI/SINet-V2.
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
| Prediction Maps | [Baidu Netdisk](https://pan.baidu.com/s/1DA3Sat-NnsNxFfN59cVLIw?pwd=digs) | digs |

---
通过网盘分享的文件：FCHNET.zip
链接:  提取码: 
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

If you have any questions, please feel free to contact me via email at carbonsir@126.com.
