# Transformer Model Translator

**Transformer Model Translator** is a Python project designed to translate text between languages using transformer-based models. The project provides flexibility to integrate with popular transformer architectures (e.g., BERT, GPT, T5) for language translation tasks.

## Features

- Transformer-based language translation
- Configurable language pairs and hyperparameters
- Flexible model integration (support for Hugging Face, etc.)
- Scalable for different languages and tasks
- Easy dependency management with Poetry

## Requirements

- Python 3.6+
- Poetry (as the dependency manager)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/transformer-model-translater.git
    cd transformer-model-translater
    ```
2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```
3.  Activate the environment:
    ```bash
    poetry shell
    ```
4. Run the main to start training

## Configuration

The project uses a configuration file, `config.yaml`, to define key parameters for the translation task. This includes:

- **Source and target languages**: Defines the language pair for translation (e.g., English to French).
- **Batch size**: The number of samples processed in each training/testing step.
- **Learning rate**: The step size for model optimization.
- **Number of epochs**: The number of times the model will iterate over the training dataset.


