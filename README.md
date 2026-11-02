# BRIDGE
# A Scalable Registration Framework for Terabyte-Scale Multi-Round Whole-Brain Microscopy
BRIDGEA Scalable Registration Framework for Terabyte-Scale Multi-Round Whole-Brain MicroscopyThis is the official PyTorch implementation of BRIDGE, a deep learning framework designed for terabyte-scale volumetric microscopic image registration (e.g., VISoR imaging). BRIDGE combines SwinB-Net—a hybrid CNN-Transformer architecture with a hierarchical 3D Swin Transformer bottleneck and a sparse B-spline displacement head—to achieve sub-cellular precision while maintaining rigorous topological regularity.

## Results
# Experimental results of the dataset：
<img width="1200" height="1100" alt="fig2" src="https://github.com/user-attachments/assets/d1cb7dab-8e2d-46f0-b172-2fdafc3fdb32" />

The multi-round registration results of all brain cell nuclei
<img width="1200" height="1100" alt="fig3" src="https://github.com/user-attachments/assets/3c53e587-90e4-4835-8e24-bc89550e7294" />

The table for calculating time is as follows
<img width="624" height="164" alt="calc_tabel" src="https://github.com/user-attachments/assets/09981cfa-cdde-4a4c-8805-421f7b1f80ec" />
<img width="199" height="691" alt="calc_time_tabel" src="https://github.com/user-attachments/assets/686563fb-9d71-4963-b486-a7192b6e3a85" />

## Citation
```bash
@article{bridge2026,
  title={BRIDGE: Scalable and Topologically Plausible Whole-Brain Microscopic Image Registration via Hierarchical Hybrid Transformers},
  author={Debin Xia, oE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, Anhui Province Key Laboratory of Biomedical Imaging and Intelligent Processing, Institute of Artificial Intelligence of Hefei Comprehensive National Science Center, National Engineering Laboratory for Brain-inspired Intelligence Technology and Application, Hefei National Research Center for Physical Sciences at the Microscale, University of Science and Technology of China, Hefei, China},
  journal={In submission},
  year={2026}
}
```

## Installation

```bash
# Clone the repository
git clone [[https://github.com/YourUsername/BRIDGE.git](https://github.com/YourUsername/BRIDGE.git)](https://github.com/mesobrain/BRIDGE)
cd BRIDGE

# Create conda environment
conda create -n bridge python=3.9 -y
conda activate bridge

# Install PyTorch (adjust CUDA version to match your hardware)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install dependencies
pip install -r requirements.txt

```

## Quick Start
1.Multi-Stage Whole-Brain Registration
Run patch-wise registration or evaluation on the cellular benchmark:
```bash
python BRIDGE.py \
    --config configs/inference_benchmark.yaml \
    --checkpoint checkpoints/swinb_net_best.pth \
    --input_dir data/nuclei_validation \
    --output_dir results/nuclei_eval
```

2. Whole-Brain Registration with multi-channels transform
Run patch-wise registration or evaluation on the cellular benchmark which adding "other channel":
```bash
python BRIDGE.py \
    --config configs/inference_benchmark.yaml \
    --checkpoint checkpoints/swinb_net_best.pth \
    --input_dir data/nuclei \
    --output_dir results/nuclei_brain \
    --other_channels data/channels
```

3. Model Training
You can start training with the default configurations, or customize hyperparameters and paths via command-line arguments.
```bash
python BRIDGE.py \
    --config configs/inference_benchmark.yaml \
    --checkpoint checkpoints/swinb_net_best.pth \
    --input_dir data/nuclei \
    --output_dir results/nuclei_brain \
    --other_channels data/channels
```
**Basic run (uses default relative paths and parameters):**
```bash
python Registration/Baser/Train.py \
    --name "swinb_experiment_01" \
    --device "cuda:0" \
    --batch_size 2 \
    --lr 0.0001 \
    --epochs 500 \
    --data_dir "/path/to/your/dataset" \
    --model_dir "/path/to/save/checkpoints"
```
4. Model Evaluation
Run inference on a pre-processed benchmark dataset, evaluating registration accuracy (NCC) and saving the warped images and affine fields.

**Basic run (uses default relative paths and parameters):**
```bash
python Registration/Predict/Prediction.py \
    --name "mcanet_bspl_down1_cnn7" \
    --device "cuda:1" \
    --model_epoch 100 \
    --batch_size 2 \
    --downsample_stride 1 \
    --kernel_size 7 \
    --model_dir "/path/to/save/checkpoints" \
    --data_dir "/path/to/Validation/WT2_R123_256_v2_nor" \
    --save_dir "/path/to/Output/predict_output"
```

## Data and Pretrained Checkpoints
Due to volume constraints, sample benchmarks and pretrained model checkpoints are archived on Zenodo:
Download Link: Zenodo Repository (DOI: [10.5281/zenodo.22895154](https://zenodo.org/records/22895154))
