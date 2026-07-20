# Zero-shot Diffusive Image Restoration with Consistency

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

Official PyTorch implementation of **"Zero-shot Diffusive Image Restoration with Consistency"**.

DPGTL is a zero-shot image restoration framework that uses pretrained diffusion
models as image priors and enforces consistency with the observed degraded
measurement during sampling. It supports multiple inverse imaging problems in a
unified framework without task-specific training or fine-tuning.

> The paper link：https://www.sciencedirect.com/science/article/abs/pii/S0165168426002197

## Highlights

- **Zero-shot restoration:** no task-specific training or fine-tuning is required.
- **Consistency-guided sampling:** restored images are constrained by the input
  measurement and its degradation model.
- **Unified interface:** the same entry point supports several image restoration
  tasks and noise settings.
- **Flexible operators:** both an SVD-based implementation and a simplified
  operator-based implementation are provided.
- **Evaluation included:** PSNR, SSIM, and LPIPS are reported by the evaluation
  pipeline.

## Supported Tasks

| Task | `--deg` value | Main parameter |
| --- | --- | --- |
| Super-resolution (average pooling) | `sr_averagepooling` | `--deg_scale` |
| Super-resolution (bicubic) | `sr_bicubic` | `--deg_scale` |
| Denoising | `denoising` | `--sigma_y` |
| Colorization | `colorization` | — |
| Inpainting | `inpainting` | mask in `exp/inp_masks/` |
| Uniform deblurring | `deblur_uni` | — |
| Gaussian deblurring | `deblur_gauss` | — |
| Anisotropic deblurring | `deblur_aniso` | — |
| Walsh–Hadamard compressed sensing | `cs_walshhadamard` | `--deg_scale` |
| Block-based compressed sensing | `cs_blockbased` | `--deg_scale` |
| Old-photo restoration | `mask_color_sr` | `--deg_scale`, `--sigma_y` |
| User-defined degradation | `diy` | edit the operator in `guided_diffusion/diffusion.py` |

The simplified implementation supports operator-based restoration tasks. Run
without `--simplified` to use the SVD-based implementation, which additionally
supports the full set of matrix-based degradations listed above.

## Repository Structure

```text
DPGTL/
├── configs/                 # Dataset, model, and sampling configurations
├── datasets/                # Dataset loaders and preprocessing
├── exp/
│   ├── datasets/            # Input images (replace with your own data)
│   └── inp_masks/           # Inpainting masks
├── functions/               # SVD operators, sampling utilities, checkpoints
├── guided_diffusion/        # Diffusion model and restoration implementation
├── hq_demo/                 # Arbitrary-size/high-resolution demonstrations
├── evaluation.sh            # Evaluation commands
└── main.py                  # Main entry point
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wy2110317529/DPGTL.git
cd DPGTL
```

### 2. Create an environment

Python 3.8 or later is recommended. Create an isolated environment first:

```bash
conda create -n dpgtl python=3.8 -y
conda activate dpgtl
```

Install a PyTorch build compatible with your CUDA driver by following the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
pip install numpy scipy pillow pyyaml tqdm requests blobfile tensorboard \
    torchvision scikit-image lpips kornia matplotlib
```

The code automatically uses a CUDA device when one is available. A CUDA-capable
GPU is strongly recommended for diffusion sampling.

## Pretrained Models

Model weights are not stored in this repository. The main program downloads the
required checkpoint automatically on first use and saves it under `exp/logs/`.

The current implementation uses the following pretrained diffusion models:

- **CelebA-HQ:** the checkpoint used by
  [SDEdit](https://github.com/ermongroup/SDEdit), stored as
  `exp/logs/celeba/celeba_hq.ckpt`.
- **ImageNet 256×256:** the unconditional model released with
  [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion), stored
  as `exp/logs/imagenet/256x256_diffusion_uncond.pt`.

If automatic downloading is unavailable, download the ImageNet checkpoint
manually:

```text
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

and place it at:

```text
exp/logs/imagenet/256x256_diffusion_uncond.pt
```

Please follow the original model licenses and terms of use.

## Data Preparation

Place an input dataset in a subdirectory of `exp/datasets/`:

```text
exp/datasets/
└── my_images/
    ├── image_001.png
    ├── image_002.jpg
    └── ...
```

Pass the subdirectory name through `--path_y my_images`. The sample data in this
repository is intended only to demonstrate the expected directory layout. Users
are responsible for complying with the licenses and terms of their datasets.

## Quick Start

### Face image super-resolution

Run 4× average-pooling super-resolution with the CelebA-HQ diffusion prior:

```bash
python main.py --ni --simplified \
    --config celeba_hq.yml \
    --path_y celeba_hq \
    --eta 0.85 \
    --deg sr_averagepooling \
    --deg_scale 4 \
    --sigma_y 0 \
    -i demo_sr4
```

Results are written to:

```text
exp/image_samples/demo_sr4/
```

### General image super-resolution

```bash
python main.py --ni --simplified \
    --config imagenet_256.yml \
    --path_y imagenet \
    --eta 0.85 \
    --deg sr_averagepooling \
    --deg_scale 4 \
    --sigma_y 0 \
    -i imagenet_sr4
```

### Noisy restoration

Use `--add_noise` when the program should synthesize noisy measurements:

```bash
python main.py --ni \
    --config celeba_hq.yml \
    --path_y celeba_hq \
    --eta 0.85 \
    --deg sr_averagepooling \
    --deg_scale 4 \
    --sigma_y 0.1 \
    --add_noise \
    -i celeba_sr4_noisy
```

## Command-Line Options

The general command is:

```bash
python main.py --ni [--simplified] \
    --config CONFIG.yml \
    --path_y INPUT_DIRECTORY \
    --eta ETA \
    --deg DEGRADATION \
    --deg_scale SCALE \
    --sigma_y NOISE_LEVEL \
    -i OUTPUT_NAME
```

Important arguments:

- `--config`: configuration filename under `configs/`.
- `--path_y`: input directory name under `exp/datasets/`.
- `--deg`: degradation or restoration task.
- `--deg_scale`: scale or sampling ratio used by the selected degradation.
- `--sigma_y`: standard deviation of the measurement noise.
- `--eta`: stochasticity parameter used during sampling; the default is `0.85`.
- `--simplified`: use the operator-based implementation without SVD.
- `--add_noise`: synthesize noise with the specified `--sigma_y`.
- `--seed`: random seed; the default is `1234`.
- `-i`, `--image_folder`: output directory name under `exp/image_samples/`.

Sampling speed and quality can be controlled in the YAML configuration:

```yaml
time_travel:
  T_sampling: 100
  travel_length: 1
  travel_repeat: 1
```

## Reproducing the Experiments

Prepare the CelebA-HQ and ImageNet evaluation data under `exp/datasets/`, then
run:

```bash
bash evaluation.sh
```

The script contains commands for super-resolution, deblurring, colorization,
compressed sensing, and inpainting. Adjust `sampling.batch_size` and the
time-travel parameters in the corresponding configuration when memory or runtime
is limited.

## High-Resolution Demo

The `hq_demo/` directory contains the arbitrary-size/high-resolution restoration
pipeline. Download the Guided Diffusion checkpoints:

```text
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
```

Place them in `hq_demo/data/pretrained/`, then run:

```bash
cd hq_demo
bash evaluation.sh
```

High-resolution diffusion sampling can require substantial GPU memory and may
take a long time. Reduce the timestep or jump parameters in `hq_demo/confs/` if
necessary.

## Citation

If this work is useful for your research, please cite the paper. The final
BibTeX entry will be added when the paper metadata is publicly available.

## Acknowledgements

This repository is developed on top of
[DDNM](https://github.com/wyhuai/DDNM). It also contains or is inspired by code
from [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion),
[RePaint](https://github.com/andreas128/RePaint),
[DDRM](https://github.com/bahjat-kawar/ddrm), and
[SDEdit](https://github.com/ermongroup/SDEdit). We thank the authors for making
their code and pretrained models available.

## License

The DDNM-derived code is distributed under the MIT License; see
[`LICENSE.txt`](LICENSE.txt) and retain the original copyright and permission
notice.

Parts of `hq_demo/` contain upstream code marked as **CC BY-NC-SA 4.0** and for
academic research use only. Those files remain subject to their own copyright
headers and license terms. Third-party pretrained models and datasets are also
subject to their respective licenses and terms of use.

