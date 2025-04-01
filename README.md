# ReferSplat

---
## Abstract
We introduce Referring 3D Gaussian Splatting
Segmentation (R3DGS), a new task that focuses
on segmenting target objects in a 3D Gaussian
scene based on natural language descriptions.
This task requires the model to identify newly
described objects that may be occluded or not
directly visible in a novel view, posing a significant challenge for 3D multi-modal understanding. Developing this capability is crucial for advancing embodied AI. To support research in this
area, we construct the first R3DGS dataset, **RefLERF**. Our analysis reveals that 3D multi-modal
understanding and spatial relationship modeling
are key challenges for R3DGS. To address these
challenges, we propose **ReferSplat**, a framework
that explicitly models 3D Gaussian points with
natural language expressions in a spatially aware
paradigm. ReferSplat achieves state-of-the-art
performance on both the newly proposed R3DGS
task and 3D open-vocabulary segmentation benchmarks. Code, trained models, and the dataset will
be publicly released.
![ReferSplat Example](teaser.png)
## Datasets
The **RefLERF dataset** is accessible for download via the following link:https://pan.baidu.com/s/1D9yDNfUrK-d8eGO33Avkpg?pwd=ugs3
```bash
<path to dataset>
|---figurines
|---ramen
|---waldo_kitchen
|---teatime
```

## Cloning the Repository
Our default, provided install method is based on Conda package and environment management:
```bash
git clone https://github.com/jgq111/ReferSplat.git
cd ReferSplat
```
## Setup
```bash
conda env create --file environment.yml
conda activate refsplat
```
## Training
```bash
python train.py -s <path to ref-lerf dataset> -m <path to output_model>
<ref-lerf>
|---data
|   |---<figurines>
|   |---<ramen>
|   |---...
|---output
    |---<figurines>
    |---<ramen>
    |---...
```

## Render
```bash
python render.py -m <path to output_model>
```
