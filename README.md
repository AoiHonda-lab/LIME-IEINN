<img src="写真URL" width="サイズ">
# LIME + IEINN

This repository provides an extended version of LIME that replaces the linear surrogate model with an interpretable neural network called **IEINN** (Interaction-Enhanced Interpretable Neural Network).

This implementation is designed to improve the explanation quality of LIME by capturing non-linear interactions between features.

## Structure

- `lime_ieinn/`: Core implementation of LIME with IEINN support
- `examples/`: Example notebooks demonstrating usage on tabular and image data

## Installation

```bash
git clone https://github.com/yourusername/lime_ieinn.git
cd lime_ieinn
pip install -e .
```

## Usage
