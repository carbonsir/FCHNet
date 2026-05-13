## 📊 Dataset Preparation

FCHNet is trained and evaluated on commonly used camouflaged object detection benchmarks:

- CAMO
- COD10K
- NC4K

The training and testing datasets can be downloaded from publicly available COD dataset repositories, such as SINet-V2:

```text
https://github.com/GewelsJI/SINet-V2
```

Please organize the dataset as follows:

```text
dataset/
├── TrainDataset/
│   ├── Image/
│   ├── GT/
│   └── Edge/
│
└── TestDataset/
    ├── CAMO/
    │   ├── Image/
    │   └── GT/
    ├── COD10K/
    │   ├── Image/
    │   └── GT/
    └── NC4K/
        ├── Image/
        └── GT/
```

The `Edge/` folder contains edge maps generated from ground-truth masks. download:  https://drive.google.com/file/d/1oxqxjQbDOcwREq53q3cU4zkdfZ6azKPK/view?usp=drive_link
