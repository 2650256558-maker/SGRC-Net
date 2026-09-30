# SGRC-Net
SGRC-Net
Official implementation of SGRC-Net (Selective Graph Refinement and Relation Correction Network) for multi-label remote sensing scene classification.
Overview
SGRC-Net combines CNN-based visual representation with graph-based relation modeling. The framework contains two main components:
- SGLR-GNN: performs global relation modeling and demand-driven sparse local refinement.
- ASRC: adaptively combines CNN and GNN predictions and applies selective relation correction at the label level.
Experiments are conducted on AID-Multilabel, DFC15-Multilabel, and MLRSNet.
Project Structure
SGRC-Net/
├── models/          # Model definitions
├── dataset.py       # Dataset loading and preprocessing
├── main.py          # Main entry point
├── metrics.py       # Evaluation metrics
├── train.py         # Training entry point
├── test.py          # Testing entry point
├── trainer.py       # Training / validation / testing pipeline
├── utils.py         # Utility functions
└── README.md
Requirements
The code is implemented with Python and PyTorch. Main dependencies include:
- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas
- tqdm
Install the required packages according to your local CUDA and PyTorch environment.
Dataset Preparation
Prepare the datasets under the configured dataset root directory and keep their training, validation, and testing splits consistent with the experimental settings.
Supported datasets:
- AID-Multilabel
- DFC15-Multilabel
- MLRSNet
Training
python train.py --dataset AID-Multilabel
Other datasets can be selected by changing the --dataset argument.
Testing
python test.py --dataset AID-Multilabel
The test script loads the corresponding trained checkpoint according to the configured evaluation protocol.
Notes
- The default backbone is ResNet-101 with ImageNet pretrained weights.
- Training and evaluation settings can be configured through command-line arguments in main.py.
- For reproducible experiments, keep the random seed, dataset split, and training configuration consistent.
License
This repository is intended for academic research and educational use.
