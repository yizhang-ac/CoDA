# CoDA: Co-Adaptive Dual-Path Alignment for Vision-Language Models

Official implementation of **CoDA: Co-Adaptive Dual-Path Alignment for Vision-Language Models**, accepted by **IJCAI-ECAI 2026**.

**Authors:** Yi Zhang, Rui Zhu, Channi Li, Xiaoxu Li, Zhanyu Ma, and Jing-Hao Xue

## Introduction

Parameter-efficient adaptation is a practical way to transfer pre-trained vision-language models such as CLIP to downstream tasks. However, under limited supervision, overly aggressive cross-modal alignment can distort modality-specific representation structures and weaken generalization.

CoDA addresses this problem by coordinating two complementary objectives:

- **Cross-modal semantic alignment**, which enables useful semantic information to be exchanged between the image and text branches.
- **Intra-modal structural consistency**, which preserves the representation structure learned during large-scale pre-training.

CoDA contains three main components:

1. **Decoupled Squeeze-and-Excitation (dcSE):** enables controlled channel-level interaction through private and shared semantic responses.
2. **Distribution-Structure Co-adaptation (DiSCo):** combines a distribution-aware `beta` path for cross-modal semantic bias with a structure-aware `gamma` path for intra-modal calibration.
3. **Parent Class Generation (PCG):** uses LLM-generated parent concepts to introduce coarse-to-fine semantic priors into textual prompts.

Experiments on 11 benchmark datasets show that CoDA performs strongly in base-to-new generalization, few-shot learning, and domain-shift settings.

## Main Results

All results below use a CLIP ViT-B/16 backbone.

| Evaluation setting | Metric | CoDA |
| --- | --- | ---: |
| Base-to-new generalization | Average base accuracy | 85.20 |
| Base-to-new generalization | Average novel accuracy | 77.44 |
| Base-to-new generalization | Average harmonic mean | **81.13** |
| Few-shot learning | 1-shot average accuracy | 72.5 |
| Few-shot learning | 4-shot average accuracy | 77.9 |
| Few-shot learning | 16-shot average accuracy | **83.4** |
| Domain generalization | Average accuracy | **60.97** |

The evaluation covers ImageNet, Caltech101, OxfordPets, StanfordCars, OxfordFlowers, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, and UCF101.

## GPU Memory Requirements

All experiments can be run on a single GPU. For ImageNet experiments, the GPU should have **more than 24 GB of memory**. Approximately **12 GB** is sufficient for the other datasets.

## How to Install

CoDA is implemented on top of a lightly modified [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) training framework. Prepare the environment as follows.

### Create a conda environment

```bash
conda create -n coda python=3.9
```

### Activate the environment

```bash
conda activate coda
```

### Install dependencies

Run the following commands from the CoDA repository root:

```bash
pip install -r requirements.txt

# Install PyTorch (version >= 1.7.1) and torchvision.
# Please install a GPU-enabled build for efficient training.
# Example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install the library in development mode.
# Reinstallation is not required after modifying the source code.
python setup.py develop
```

The packages required by [CLIP](https://github.com/openai/CLIP) are included in `requirements.txt`. Make sure the `coda` environment is active when installing them.

## Dataset Preparation

Follow [DATASETS.md](DATASETS.md) to download and organize the datasets. Ensure that the dataset paths in the relevant configuration files match your local directory structure before starting an experiment.

## How to Run

### Generalization from Base to New Classes

Run the training script in the foreground:

```bash
cd scripts
bash base2new_train.sh
```

To run it in the background and save the output to a log file:

```bash
cd scripts
nohup bash base2new_train.sh > base2new_train.log 2>&1 &
```

## Citation

If you find CoDA useful in your research, please cite our paper:

```bibtex
@inproceedings{zhang2026coda,
  title     = {CoDA: Co-Adaptive Dual-Path Alignment for Vision-Language Models},
  author    = {Zhang, Yi and Zhu, Rui and Li, Channi and Li, Xiaoxu and Ma, Zhanyu and Xue, Jing-Hao},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Acknowledgements

This project builds on [CLIP](https://github.com/openai/CLIP) and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). We thank the authors of these projects for making their code publicly available.

## Contact

For questions about CoDA, please contact:

- Yi Zhang: `yizhang.cv.ac@gmail.com`
- Xiaoxu Li: `lixiaoxu@lut.edu.cn`
- Zhanyu Ma: `mazhanyu@bupt.edu.cn`

Project repository: <https://github.com/yizhang-ac/CoDA>
