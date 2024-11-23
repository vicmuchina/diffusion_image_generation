import torch  # PyTorch: Deep learning framework
import diffusers  # HuggingFace Diffusers: Library for diffusion models
from datasets import load_dataset  # HuggingFace Datasets: For loading datasets
from torchvision import transforms  # For image transformations
import lightning as L  # PyTorch Lightning: High-level training framework


class DiffusionModel(L.LightningModule):
    """
    A diffusion model implementation using PyTorch Lightning.
    
    This model implements a U-Net architecture for image generation using the DDPM
    (Denoising Diffusion Probabilistic Model) approach. The model learns to denoise
    images by predicting the noise added at different timesteps.
    """
    def __init__(self):
        super().__init__()
        # Initialize U-Net model with specific architecture
        # - block_out_channels: Number of channels in each block
        # - down/up_block_types: Types of blocks for encoder/decoder paths
        # - sample_size: Input image size (32x32)
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            sample_size=32,
        )
        # Initialize DDPM scheduler for noise addition/removal
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            variance_type="fixed_large",  # Type of variance schedule
            clip_sample=False,  # Whether to clip samples during training
            timestep_spacing="trailing",  # How timesteps are spaced
        )

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        
        Process:
        1. Get images from batch
        2. Generate random noise
        3. Add noise to images at random timesteps
        4. Predict the noise using the model
        5. Calculate MSE loss between predicted and actual noise
        """
        images = batch["images"]
        noise = torch.randn_like(images)  # Generate random noise
        # Sample random timesteps for each image
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        # Add noise to images according to the timesteps
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # Predict the noise
        residual = self.model(noisy_images, steps).sample
        # Calculate loss
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        Uses AdamW optimizer with learning rate 1e-4 and StepLR scheduler
        that reduces learning rate by 1% every epoch.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    """
    DataModule for handling the CIFAR10 dataset.
    Implements data loading and preprocessing logic.
    """
    def __init__(self):
        super().__init__()
        # Define image transformations
        self.augment = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize to 32x32
            transforms.CenterCrop(32),  # Ensure exact size
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if not already present."""
        load_dataset("cifar10")

    def train_dataloader(self):
        """
        Creates the training data loader.
        Returns a DataLoader with CIFAR10 images, batch size 128,
        shuffled, using 4 worker processes.
        """
        dataset = load_dataset("cifar10")
        # Apply transformations to the images
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=4)


if __name__ == "__main__":
    # Initialize model, data, and trainer
    model = DiffusionModel()
    data = DiffusionData()
    # Train for 150 epochs using bfloat16 mixed precision
    trainer = L.Trainer(max_epochs=150, precision="bf16-mixed")
    trainer.fit(model, data)
