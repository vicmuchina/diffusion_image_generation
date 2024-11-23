# Diffusion Image Generation

![Project Logo](https://your-project-logo-url.com/logo.png)

## Table of Contents

1. [Overview](#overview)
2. [Concepts](#concepts)
   - [Diffusion Models](#diffusion-models)
   - [Training Process](#training-process)
   - [Image Generation](#image-generation)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Generating Images](#generating-images)
   - [Stopping and Resuming Training](#stopping-and-resuming-training)
6. [Project Structure](#project-structure)
7. [Commands](#commands)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** for image generation using PyTorch Lightning. Diffusion models are a class of generative models that learn to generate data by iteratively denoising random noise. This approach has shown impressive results in creating high-quality images.

## Concepts

### Diffusion Models

A **Diffusion Model** is a type of generative model that learns to generate data by reversing a gradual noising process. The model starts with pure noise and iteratively refines it to produce a coherent image.

### Training Process

During training, the model learns to predict the noise added to images at various timesteps. The training involves the following steps:

1. **Add Noise**: Gradually add noise to real images.
2. **Predict Noise**: Train the model to predict the added noise.
3. **Calculate Loss**: Compute the difference between predicted noise and actual noise using Mean Squared Error (MSE) loss.
4. **Optimize**: Adjust model parameters to minimize the loss.

### Image Generation

After training, image generation involves starting with random noise and iteratively denoising it using the trained model to produce a clear image.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **GPU**: (Optional but recommended) NVIDIA GPU with CUDA support for faster training

## Installation

Follow these steps to set up your development environment.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vicmuchina/diffusion_image_generation.git
   cd diffusion_image_generation
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **Windows**

     ```bash
     venv\Scripts\activate
     ```

   - **macOS and Linux**

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt` file, you can install the necessary packages manually:*

   ```bash
   pip install torch torchvision diffusers datasets pytorch-lightning jsonargparse
   ```

## Usage

### Training the Model

To train the diffusion model on the CIFAR10 dataset, run the `train.py` script.

```bash
python train.py
```

#### Customizing Training Parameters

You can customize training parameters using command-line arguments or a configuration file. For example, to change the number of epochs:

```bash
python train.py --max_epochs 100
```

*Note: Ensure you have a `config.yaml` file if using configuration files.*

### Generating Images

After training, use the `generate.py` script to create new images.

```bash
python generate.py
```

#### Customizing Generation Parameters

You can adjust the generation settings like the number of images, timesteps, and seed. For example:

```bash
python generate.py --checkpoint "lightning_logs/version_0/checkpoints/epoch=99-step=9800.ckpt" --num_timesteps 500 --num_samples 100 --seed 42
```

### Stopping and Resuming Training

#### Stopping Training Early

You can stop the training process at any time by pressing `Ctrl+C` in the terminal. PyTorch Lightning automatically saves the current state of the model as a checkpoint in the `lightning_logs` directory.

#### Resuming Training

To resume training from the last checkpoint, use the `--ckpt_path` argument:

```bash
python train.py --ckpt_path "lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt"
```

*Replace `version_X`, `epoch=Y`, and `step=Z` with your specific checkpoint details.*

## Project Structure

```
diffusion_image_generation/
├── lightning_logs/            # Directory where checkpoints and logs are saved
├── train.py                   # Script to train the diffusion model
├── generate.py                # Script to generate images using the trained model
├── generated.png              # Example of generated images
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Commands

Here are some useful commands to manage your project:

- **Activate Virtual Environment**

  - **Windows**

    ```bash
    venv\Scripts\activate
    ```

  - **macOS and Linux**

    ```bash
    source venv/bin/activate
    ```

- **Deactivate Virtual Environment**

  ```bash
  deactivate
  ```

- **Install Dependencies**

  ```bash
  pip install -r requirements.txt
  ```

- **Train the Model**

  ```bash
  python train.py
  ```

- **Generate Images**

  ```bash
  python generate.py
  ```

- **Stop Training**

  Press `Ctrl+C` in the terminal where training is running.

- **Resume Training**

  ```bash
  python train.py --ckpt_path "lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt"
  ```

## Troubleshooting

- **CUDA Errors**

  Ensure that you have a compatible NVIDIA GPU and that CUDA is properly installed. You can verify GPU availability in Python:

  ```python
  import torch
  print(torch.cuda.is_available())
  ```

- **Missing Dependencies**

  If you encounter errors related to missing packages, ensure all dependencies are installed:

  ```bash
  pip install -r requirements.txt
  ```

- **Insufficient GPU Memory**

  If your GPU runs out of memory, consider reducing the `batch_size` in `train.py`:

  ```python
  return torch.utils.data.DataLoader(
      dataset["train"], 
      batch_size=64,  # Reduced from 128
      shuffle=True, 
      num_workers=4
  )
  ```

- **Checkpoints Not Saving**

  Ensure the `lightning_logs` directory exists and has write permissions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Changes and Commit**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

---

## Additional Notes

- **Security Best Practices**

  **Warning:** Avoid committing sensitive information such as API keys or credentials to your Git repository. In your Git commands, ensure that no sensitive data is exposed. Use environment variables or `.gitignore` to protect sensitive files.

- **Performance Optimization**

  - Utilize GPUs for faster training. Ensure CUDA drivers are installed and configured correctly.
  - Adjust `precision` in `train.py` to balance between speed and accuracy. Mixed precision (`bf16-mixed`) can speed up training with minimal loss in model performance.

- **Monitoring Training Progress**

  PyTorch Lightning provides progress bars and logging to monitor training. Additionally, you can integrate TensorBoard or other visualization tools for more detailed insights.

regards Lightning
--- 