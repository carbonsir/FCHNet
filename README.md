# FCHNet: Frequency-aware Cross-domain Hierarchical Network for Camouflaged Object Detection
> 🚧 **Note**  
> The code and pretrained models will be released after the paper is accepted for publication.

## 📌 Abstract
<p align="justify">
Camouflaged object detection (COD) remains challenging due to the high visual similarity between camouflaged objects and their surroundings, which often obscures subtle structures and weak boundaries. Recent COD methods incorporate frequency-domain priors to enhance structural representation. However, they typically rely on loosely coupled fusion, which fails to maintain consistent coordination between priors and evolving feature representations during hierarchical processing. To address this issue, we propose a Frequency-aware Cross-domain Hierarchical Network (FCHNet), which explicitly models the interaction between image-derived frequency priors and feature representations through structured alignment. At the core of the framework, a Cross-domain Frequency Alignment (CFA) module establishes bidirectional interaction and feedback mechanisms to ensure stable prior-feature coordination across domains. To support effective alignment, a Wavelet Feature Refinement (WFR) module performs structured multi-scale decomposition and band-specific enhancement to capture subtle structural and texture cues. In addition, a boundary-aware decoding strategy progressively incorporates boundary priors to improve boundary quality and object completeness. Extensive experiments show that FCHNet outperforms existing lightweight methods by up to 3.1% in F<sub>β</sub><sup>ω</sup> on COD10K, while using less than 5M parameters, and achieves performance comparable to heavyweight models with significantly lower computational cost.
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
