import torch  # Imports the main PyTorch library for tensor operations and neural networks
import diffusers  # Imports Hugging Face's Diffusers library for building diffusion models
from datasets import load_dataset  # Imports the function to load datasets easily
from torchvision import transforms  # Imports tools for image transformations
import lightning as L  # Imports PyTorch Lightning for simplifying training loops


class DiffusionModel(L.LightningModule):
    """
    This class defines the diffusion model used for generating images.
    It inherits from PyTorch Lightning's LightningModule to utilize its features.
    
    The model uses a U-Net architecture to learn how to denoise images step by step.
    """
    def __init__(self):
        super().__init__()  # Calls the constructor of the parent LightningModule
        # Initializes the U-Net model with specific configurations
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),  # Number of output channels for each block
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", 
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),  # Types of blocks used in the downsampling path
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
                "UpBlock2D", "UpBlock2D", "UpBlock2D"
            ),  # Types of blocks used in the upsampling path
            sample_size=32,  # Size of the input images (32x32 pixels)
        )
        # Initializes the scheduler responsible for adding and removing noise
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            variance_type="fixed_large",  # Type of variance schedule used in the scheduler
            clip_sample=False,  # Whether to clip the sample values to a specific range
            timestep_spacing="trailing",  # How timesteps are spaced during denoising
        )

    def training_step(self, batch, batch_idx):
        """
        Defines a single step during training.
        
        Args:
            batch: A batch of data containing images.
            batch_idx: Index of the batch.
        
        Process:
            1. Extract images from the batch.
            2. Generate random noise matching the images' shape.
            3. Select random timesteps for each image.
            4. Add noise to the images based on the selected timesteps.
            5. Use the model to predict the added noise.
            6. Calculate the Mean Squared Error (MSE) loss between predicted noise and actual noise.
            7. Log the training loss for monitoring.
        
        Returns:
            The calculated loss for backpropagation.
        """
        images = batch["images"]  # Extracts images from the batch
        noise = torch.randn_like(images)  # Generates random noise with the same shape as images
        # Selects random timesteps for each image in the batch
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps, 
            (images.size(0),), 
            device=self.device
        )
        # Adds noise to the images based on the selected timesteps
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # Uses the model to predict the residual (noise) added to the images
        residual = self.model(noisy_images, steps).sample
        # Calculates the Mean Squared Error (MSE) loss between predicted noise and actual noise
        loss = torch.nn.functional.mse_loss(residual, noise)
        # Logs the training loss to the progress bar for monitoring
        self.log("train_loss", loss, prog_bar=True)
        return loss  # Returns the loss for backpropagation

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.
        
        Returns:
            A tuple containing the optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)  # Initializes AdamW optimizer with learning rate 0.0001
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)  # Scheduler reduces LR by 1% every epoch
        return [optimizer], [scheduler]  # Returns optimizer and scheduler

    
class DiffusionData(L.LightningDataModule):
    """
    This class handles all data-related operations, such as downloading, processing, and loading data.
    It inherits from PyTorch Lightning's LightningDataModule for streamlined data management.
    """
    def __init__(self):
        super().__init__()  # Calls the constructor of the parent LightningDataModule
        # Defines a series of transformations to apply to the images
        self.augment = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),  # Resizes images to 32x32 pixels using bilinear interpolation
            transforms.CenterCrop(32),  # Crops the center of the image to ensure exact size
            transforms.RandomHorizontalFlip(),  # Randomly flips images horizontally for data augmentation
            transforms.ToTensor(),  # Converts images to PyTorch tensors
            transforms.Normalize([0.5], [0.5]),  # Normalizes image tensor values to the range [-1, 1]
        ])

    def prepare_data(self):
        """
        Downloads the CIFAR10 dataset if it's not already available.
        This method is called only from a single GPU in distributed settings.
        """
        load_dataset("cifar10")  # Downloads the CIFAR10 dataset from Hugging Face Datasets

    def train_dataloader(self):
        """
        Prepares the training data loader.
        
        Returns:
            A DataLoader object that provides batches of processed images.
        """
        dataset = load_dataset("cifar10")  # Loads the CIFAR10 dataset
        # Applies the defined transformations to each sample in the dataset
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        # Returns a DataLoader that yields batches of 128 images, shuffled and using 4 worker threads for loading
        return torch.utils.data.DataLoader(
            dataset["train"], 
            batch_size=128, 
            shuffle=True, 
            num_workers=4
        )


# Entry point for the training script
if __name__ == "__main__":
    model = DiffusionModel()  # Instantiates the diffusion model
    data = DiffusionData()    # Instantiates the data module
    
    # Defines an early stopping callback to halt training if no improvement is seen
    early_stop_callback = L.callbacks.EarlyStopping(
        monitor='train_loss',  # Metric to monitor
        min_delta=0.00001,     # Minimum change to qualify as an improvement
        patience=10,           # Number of epochs to wait for improvement before stopping
        verbose=True,          # Enables verbosity (prints messages)
        mode='min'             # Looks for a minimum in the monitored metric
    )
    
    # Configures the trainer with specified settings and callbacks
    trainer = L.Trainer(
        max_epochs=150,           # Maximum number of epochs to train
        precision="bf16-mixed",   # Uses mixed precision with bfloat16 for faster training
        callbacks=[early_stop_callback]  # Adds the early stopping callback
    )
    trainer.fit(model, data)  # Starts the training process using the model and data
