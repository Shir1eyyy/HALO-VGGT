<div align="center">
<h2>⚡️ HALO-VGGT: Heterogeneity-Aware Lightweight
Online Compression Allocator for Efficient VGGT</h2>
  


**Shanghai Jiao Tong University**


Xueling Wang, Yiwen Wang, Siqi Cai, Chen Zhang, Guanghui He
</div>



## 🔭 Overview

HALO-VGGT is a **training-free, calibration-free framework** that enables efficient VGGT-style 3D reconstruction through **lightweight online sensitivity estimation and adaptive compression allocation** across attention layers and heads.


## ⚙️ Environment Setup
First, create a virtual environment using Conda, clone this repository to your local machine, and install the required dependencies.

```bash
conda create -n halovggt python=3.10
conda activate halovggt
git clone git@github.com:mystorm16/FastVGGT.git
cd halovggt
pip install -r requirements.txt
```

Then, download the VGGT checkpoint (we use the checkpoint link provided in https://github.com/facebookresearch/vggt/tree/evaluation/evaluation):
```bash
wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
```

Finally, configure the dataset path and VGGT checkpoint path. For example:
```bash
    parser.add_argument(
        "--data_dir", type=Path, default="/data/scannetv2/process_scannet"
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default="/data/scannetv2/OpenDataLab___ScanNet_v2/raw/scans",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./ckpt/model_tracker_fixed_e20.pt",
    )
```


##  Evaluation

### Random K/V Reduction

The cleaned random-reduction implementation keeps all query tokens and reduces
only keys and values in Global Attention. The first frame is kept intact, while
later frames are reduced independently. Two deterministic selection policies are
available:

- `--rand_token_select random`: seeded pseudo-random sampling (reproducible with
  `--random_seed`).
- `--rand_token_select uniform`: evenly spaced token sampling.

`--merging` is the first Global Attention block where reduction is enabled.
`--keep_ratio` is used by non-sensitive blocks. Sensitive blocks rank attention
heads with a lightweight sampled-QK score and use `--level1` through `--level4`.

```bash
# Random sampling
python eval/eval_7andN.py \
  --method random \
  --merging 0 \
  --keep_ratio 0.05 \
  --level1 0.1 --level2 0.2 --level3 0.3 --level4 0.4 \
  --rand_token_select random

# Uniform sampling with the same reduction ratios
python eval/eval_7andN.py \
  --method random \
  --merging 0 \
  --keep_ratio 0.05 \
  --level1 0.1 --level2 0.2 --level3 0.3 --level4 0.4 \
  --rand_token_select uniform
```

The same flags are shared by `eval_scannet.py`, `eval_custom.py`, and
`eval_custom_colmap.py`. Use `--method vggt` for full attention or
`--method fastvggt` for the original FastVGGT token-merging path.

