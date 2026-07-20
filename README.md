# Zero-shot Diffusive Image Restoration with Consistency

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

Official PyTorch implementation of **"Zero-shot Diffusive Image Restoration with Consistency"**.

DPGTL is a zero-shot image restoration framework that uses pretrained diffusion
models as image priors and enforces consistency with the observed degraded
measurement during sampling. The main experiments focus on colorization,
compressed sensing, Gaussian deblurring, and noisy compressed sensing, without
task-specific training or fine-tuning. The implementation also retains other
DDNM restoration operators for broader evaluation and further research.

> **Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165168426002197)

## Highlights

- **Zero-shot restoration:** no task-specific training or fine-tuning is required.
- **Consistency-guided sampling:** restored images are constrained by the input
  measurement and its degradation model.
- **Main evaluation tasks:** colorization, compressed sensing, Gaussian
  deblurring, and noisy compressed sensing.
- **Flexible operators:** both an SVD-based implementation and a simplified
  operator-based implementation are provided.
- **Evaluation included:** PSNR, SSIM, and LPIPS are reported by the evaluation
  pipeline.

## Main Evaluated Tasks

The paper reports quantitative and qualitative results for the following four
experimental settings:

| Task | Experimental setting | Command-line arguments |
| --- | --- | --- |
| Colorization | Noise-free measurement | `--deg colorization` |
| Compressed sensing | Noise-free measurement | `--deg cs_walshhadamard` or `--deg cs_blockbased` |
| Gaussian deblurring | Gaussian blur kernel | `--deg deblur_gauss` |
| Noisy compressed sensing | Compressed measurement with additive noise | `--deg cs_walshhadamard --sigma_y LEVEL --add_noise` |

## Additional Evaluated Tasks

We also conducted preliminary experiments on other restoration tasks inherited
from DDNM. The proposed method improves the restoration results on these tasks,
although the gains are generally smaller than those observed on the four main
evaluation settings. These interfaces are retained for completeness and further
research.

| Task | `--deg` value | Main parameter |
| --- | --- | --- |
| Super-resolution (average pooling) | `sr_averagepooling` | `--deg_scale` |
| Super-resolution (bicubic) | `sr_bicubic` | `--deg_scale` |
| Denoising | `denoising` | `--sigma_y` |
| Inpainting | `inpainting` | mask in `exp/inp_masks/` |
| Uniform deblurring | `deblur_uni` | — |
| Anisotropic deblurring | `deblur_aniso` | — |
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

The examples below correspond to the four main experimental settings. Replace
`imagenet` with the name of your input directory under `exp/datasets/` when
using your own data.

### Colorization

```bash
python main.py --ni \
    --config imagenet_256.yml \
    --path_y imagenet \
    --eta 0.85 \
    --deg colorization \
    --sigma_y 0 \
    -i imagenet_colorization
```

Results are written to:

```text
exp/image_samples/imagenet_colorization/
```

### Compressed sensing

```bash
python main.py --ni \
    --config imagenet_256.yml \
    --path_y imagenet \
    --eta 0.85 \
    --deg cs_walshhadamard \
    --deg_scale 0.25 \
    --sigma_y 0 \
    -i imagenet_cs_wh_025
```

### Gaussian deblurring

```bash
python main.py --ni \
    --config imagenet_256.yml \
    --path_y imagenet \
    --eta 0.85 \
    --deg deblur_gauss \
    --sigma_y 0 \
    -i imagenet_deblur_gauss
```

### Noisy compressed sensing

Use `--add_noise` to synthesize measurement noise. The following example uses a
noise level of `0.2`; change it to match the setting being evaluated.

```bash
python main.py --ni \
    --config imagenet_256.yml \
    --path_y imagenet \
    --eta 0.85 \
    --deg cs_walshhadamard \
    --deg_scale 0.25 \
    --sigma_y 0.2 \
    --add_noise \
    -i imagenet_cs_wh_025_noisy
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

Prepare the CelebA-HQ and ImageNet evaluation data under `exp/datasets/`. The
main paper experiments cover colorization, compressed sensing, Gaussian
deblurring, and noisy compressed sensing. The evaluation script also retains
additional DDNM-compatible experiments:

```bash
bash evaluation.sh
```

Use the task-specific commands and the same configuration, noise level, sampling
ratio, and random seed reported in the paper when reproducing its quantitative
results. Other commands in `evaluation.sh` are supplementary experiments rather
than the main evaluation settings. Adjust `sampling.batch_size` and the
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
