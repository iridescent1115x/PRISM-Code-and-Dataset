# PRISM Code and Dataset

This repository provides the official implementation of the PRISM model, including training, evaluation, and dataset preparation instructions.

---

## Overview

This repository includes:

- Implementation of the PRISM model
- Training and evaluation scripts
- Configuration files for reproducible experiments
- Utilities for multimodal recommendation learning
- Dataset access and preparation instructions

The codebase is organized to support reproducibility of the experimental results reported in the paper.

---

## Repository Structure

- `main.py` : Entry point for training and evaluation
- `prism/` : Core implementation of the PRISM model
- `configs/` : Model and dataset configuration files
- `common/`, `utils/` : Supporting modules and training utilities
- `data/` : Dataset directory to be populated after download

---

## Dataset

The datasets used in this study are publicly available processed multimodal recommendation datasets.

Processed datasets for PRISM can be accessed at:

https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG

After downloading, please place the processed files under the `data/` directory using dataset-specific subdirectories, for example:

```text
data/
  baby/
  sports/
  clothing/
  elec/
```

The expected files for each dataset are determined by the configuration files in `configs/dataset/` and may include processed interaction data, image features, text features, and mapping files.

Additional dataset placement notes are provided in `data/README.md`.

---

## Environment

We provide a reproducible GPU environment configuration:

```bash
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.2.1
pip install -r requirements.txt
```

---

## Running the Code

### Training and Evaluation

```bash
python main.py --gpu 0 --seed 1 --dataset baby --result_dir results --method prism
```

---

## Reproducibility

All experiments reported in the paper can be reproduced using the provided code, configuration files, and processed datasets. Random seeds are fixed for consistency across runs.

---

## License

Users must comply with the license terms of all datasets and upstream dependencies used in this repository.
