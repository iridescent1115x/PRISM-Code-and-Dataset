# Data Preparation

The processed datasets used for PRISM are publicly available at:

https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG

After downloading, place the processed files into dataset-specific subdirectories under `data/`, for example:

```text
data/
  baby/
  sports/
  clothing/
  elec/
```

The exact files required for each dataset are specified by the corresponding configuration files in `../configs/dataset/`.

In general, each dataset directory should contain the processed interaction file and multimodal feature files required by the codebase, such as:

- interaction records
- image features
- text features
- item or user mapping files when needed

Please keep the directory names and file names consistent with the configuration files before running `main.py`.
