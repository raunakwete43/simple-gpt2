# Simple GPT-2 Implementation with PyTorch Lightning

This project provides a minimal yet functional implementation of a GPT-2-like language model using PyTorch and PyTorch Lightning. It includes all components required for training, inference, and tokenization on a custom text dataset.

## Features
- **Custom GPT-2 Model**: Implements a simplified GPT-2 architecture with configurable layers, embedding size, and attention heads.
- **PyTorch Lightning Integration**: Modular training and evaluation using Lightning modules and data modules.
- **Training & Inference Scripts**: Ready-to-use scripts for model training and text generation.
- **Configurable & Extensible**: Easily adjust model and training parameters.

## Project Structure
```
├── model.py         # Simplified GPT-2 model and Lightning module
├── data.py          # Data module and dataset for text loading and batching
├── train.py         # Training script using PyTorch Lightning
├── inference.py     # Script for running inference/generation
├── input.txt        # Input file for training
```

## Getting Started

### 1. Environment Setup
Install the required Python packages:

```bash
pip install torch pytorch-lightning tiktoken
```

### 2. Data Preparation
- Prepare your training data as plain text and save it as `input.txt`.

### 3. Training the Model
Run the training script:

```bash
python train.py
```

- The script will initialize the model and data module, and start training (configurable in `train.py`).
- Model checkpoints and logs will be saved in the default PyTorch Lightning output directory.

### 4. Inference
After training, generate text using the inference script:

```bash
python inference.py
```

- The script loads a trained checkpoint and generates text samples from a prompt.

## Model Details
- **Architecture**: Multi-layer transformer with configurable number of layers, heads, and embedding size (see `model.py`).
- **Training**: Uses cross-entropy loss for next-token prediction. Optimizer is AdamW with weight decay.
- **Data**: Loads and tokenizes text, splits into train/val/test, and batches for training.

## Customization
- Adjust model hyperparameters in `model.py` or via the `GPT2Config` dataclass.
- Change training parameters (epochs, batch size, etc.) in `train.py`.
- Use your own dataset by replacing `input.txt` and updating the data path in `train.py`.

## Requirements
- Python 3.8+
- torch
- pytorch-lightning
- tiktoken

## References
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [tiktoken](https://github.com/openai/tiktoken)

