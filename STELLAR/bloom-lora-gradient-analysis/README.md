# Bloom LoRA Gradient Analysis

This project provides a framework for fine-tuning the BloomZ-560M model using Low-Rank Adaptation (LoRA). It includes tools for training, evaluating, and analyzing gradients and activations during the training process.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd bloom-lora-gradient-analysis
pip install -r requirements.txt
```

Make sure to have the necessary libraries for deep learning and data manipulation installed.

## Usage

1. **Training the Model**: Run the training script to fine-tune the BloomZ-560M model with LoRA.
   ```bash
   python src/train.py --config configs/training_config.yaml
   ```

2. **Evaluating the Model**: After training, evaluate the model's performance using:
   ```bash
   python src/evaluate.py --config configs/model_config.yaml
   ```

3. **Analyzing Gradients**: Use the Jupyter notebook for interactive analysis of gradients and activations.
   ```bash
   jupyter notebook notebooks/gradient_analysis.ipynb
   ```

## Project Structure

- `src/`: Contains the main source code for training, evaluation, and utilities.
  - `train.py`: Main training loop for the model.
  - `evaluate.py`: Evaluation script for model performance.
  - `models/`: Contains model definitions.
    - `lora_model.py`: LoRA model architecture.
  - `utils/`: Utility functions for metrics and data handling.
    - `metrics.py`: Functions for computing metrics.
    - `gradient_similarity.py`: Functions for gradient similarity analysis.
    - `data_loader.py`: Data loading and preprocessing.
  - `analysis/`: Analysis tools for visualizing and correlating gradients.
    - `visualization.py`: Functions for visualizing gradients and activations.
    - `correlation.py`: Functions for computing Pearson correlation coefficients.

- `configs/`: Configuration files for model and training settings.
- `notebooks/`: Jupyter notebooks for interactive analysis.
- `requirements.txt`: List of dependencies for the project.
- `.gitignore`: Files and directories to ignore in version control.
- `README.md`: Documentation for the project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.