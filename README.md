# Non-Intrusive Speech Intelligibility Prediction from Discrete Latent Representations
Official repository for paper "Non-Intrusive Speech Intelligibility Prediction from Discrete Latent Representations"

We predict the intelligibility of binaural speech signals by first extracting latent representations from raw audio. Then, a lightweight predictor over these latent representations can be trained. This results in improved performance over predicting on spectral features of the audio, despite the feature extractor not being explicitly trained for this task. In certain cases, a single layer is sufficient for strong correlations between the predictions and the ground-truth scores.

This repository contains:
- `vqcpc/` - Module for VQCPC model in PyTorch
- `stoi/` - Module for Small and SeqPool predictor model in PyTorch
- `data.py` - File containing various PyTorch custom datasets
- `main-vqcpc.py` - Script for VQCPC training
- `create-latents.py` - Script for generating latent dataset from trained VQCPC
- `plot-latents.py` - Script for visualizing extracted latent representations
- `main-stoi.py` - Script for STOI predictor training
- `main-test.py` - Script for evaluating models
- `compute-correlations.py` - Script for computing metrics for many models
- `checkpoints/` - trained checkpoints of VQCPC and STOI predictor models
- `config/` - Directory containing various configuration files for experiments
- `results/` - Directory containing official results from experiments
- `dataset/` - Directory containing metadata files for the dataset

All models are implemented in PyTorch. The training scripts are implemented using [ptpt](github.com/vvvm23/ptpt) - a lightweight framework around PyTorch.

## Installation
`TODO: setup instructions`

## Usage

### VQ-CPC Training
`TODO: usage instructions`

### Latent Dataset Generation
`TODO: usage instructions`

### Latent Plotting
`TODO: usage instructions`

### STOI Predictor Training
`TODO: usage instructions`

### Predictor Evaluation
`TODO: usage instructions`

## Checkpoints
`TODO: add trained checkpoints`

### Citation
`TODO: add citation`
