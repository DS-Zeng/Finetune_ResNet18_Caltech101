# ResNet-18 Fine-tuning for Caltech-101 Classification

This repository contains the implementation of transfer learning using ResNet-18 for object classification on the Caltech-101 dataset. The project demonstrates the effectiveness of fine-tuning pre-trained models compared to training from scratch.

## Project Overview

- **Model**: ResNet-18 pre-trained on ImageNet
- **Dataset**: Caltech-101 (102 classes including background)
- **Best Accuracy**: 96.21% on validation set
- **Optimization**: Hyperparameter tuning using Optuna

## Dataset

The Caltech-101 dataset contains approximately 9,000 images across 101 object categories plus one background category. Download the dataset from the [official Caltech-101 page](https://data.caltech.edu/records/mzrjq-6wc02).

### Dataset Structure
```
data/
└── caltech-101/
    └── 101_ObjectCategories/
        ├── train/
        ├── val/
        └── test/
```

## Installation

### Requirements
```bash
pip install torch torchvision tqdm optuna tensorboard scikit-learn
```

### Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/resnet18-caltech101-finetuning.git
cd resnet18-caltech101-finetuning
```

2. Download and extract the Caltech-101 dataset to the `data/` directory

3. Run the data preprocessing script:
```bash
python dataloader.py
```

## Training

### Basic Training
To train the model with default hyperparameters:

```bash
cd ResNet-18
python train_ResNet-18_caltech101.py
```

### Hyperparameter Optimization
The script includes Optuna-based hyperparameter optimization. To run optimization:

1. Uncomment the Optuna optimization section in `train_ResNet-18_caltech101.py`
2. Run the training script

### Optimal Hyperparameters
Based on our optimization, the best hyperparameters are:
- Learning Rate (new layer): 0.00801
- Fine-tune Learning Rate: 0.000957
- Batch Size: 16
- Validation Accuracy: 96.21%

## Model Architecture

The model uses ResNet-18 with the following modifications:
- Pre-trained weights from ImageNet for feature extraction layers
- Modified final fully connected layer (512 → 102 classes)
- Differential learning rates for different layer groups

## Training Configuration

- **Optimizer**: SGD with momentum=0.9
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 25
- **Data Augmentation**: Random horizontal flip, resize to 224×224, ImageNet normalization

## Results

### Performance Comparison
| Method | Validation Accuracy | Training Time |
|--------|-------------------|---------------|
| Transfer Learning | 96.21% | 25 epochs |
| From Scratch | ~85-90% | 50+ epochs |

### Key Findings
1. Transfer learning significantly outperforms training from scratch
2. Differential learning rates are crucial for effective fine-tuning
3. Smaller batch sizes (16) provide better generalization
4. Systematic hyperparameter optimization yields substantial improvements

## Model Weights

Pre-trained model weights are available for download:
- **Download Link**: [Google Drive](https://drive.google.com/your-model-weights-link)
- **File**: `resnet18_caltech101.pth` (43MB)

### Loading Pre-trained Model
```python
import torch
import torch.nn as nn
from torchvision import models

# Load model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)

# Load pre-trained weights
model.load_state_dict(torch.load('resnet18_caltech101.pth'))
model.eval()
```

## Testing

To evaluate the model on test data:

```python
# Add evaluation code here
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy
```

## Visualization

The training process can be monitored using TensorBoard:

```bash
tensorboard --logdir=runs
```

This will show:
- Training and validation loss curves
- Validation accuracy progression
- Learning rate schedules

## File Structure

```
├── README.md
├── dataloader.py              # Data preprocessing and splitting
├── resnet18_caltech101.pth   # Pre-trained model weights
├── report.tex                # LaTeX experiment report
├── ResNet-18/
│   └── train_ResNet-18_caltech101.py  # Main training script
└── data/
    └── caltech-101/          # Dataset directory
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{resnet18_caltech101_2024,
  title={Fine-tuning ResNet-18 for Caltech-101 Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/resnet18-caltech101-finetuning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet-18 architecture from torchvision
- Caltech-101 dataset from Caltech Vision Lab
- Optuna for hyperparameter optimization
- PyTorch team for the deep learning framework 