# MADFlow
MADFlow: Multimodal Difference Compensation Flow for Multimodal Anomaly Detection



## Setup
We implement this repo with the following environment:
- Ubuntu 22.04
- Python 3.8
- Pytorch 2.1.2
- CUDA 12.1

Install the other package via:
``` bash
pip install -r requirement.txt
```
## Data Download and Preprocess

### Dataset

- The `MVTec 3D-AD` dataset can be download from the [Official Website of MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

- The `Eyecandies` dataset can be download from the [Official Website of Eyecandies](https://eyecan-ai.github.io/eyecandies/).

After download, put the dataset in `dataset` folder.

## Train

```bash
python train_cma.py
python train.py
```

## Test

```bash
python eval.py
```
