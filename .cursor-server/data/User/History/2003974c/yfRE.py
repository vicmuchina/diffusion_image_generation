import torch  # Main deep learning library - helps us work with neural networks
import diffusers  # Library that provides tools for creating AI image generators
from datasets import load_dataset  # Tool to easily download and use common datasets
from torchvision import transforms  # Tools for processing and modifying images
import lightning as L  # Makes it easier to organize and train AI models


class DiffusionModel(L.LightningModule):
    """
    This is our main AI model that will learn to create images.
    It works by learning how to clean up noisy images step by step.
    Think of it like learning to clean a dirty window - you start with
    a very dirty window and gradually clean it until you can see through it clearly.
    """
    def __init__(self):
        super().__init__()
        # Create the main neural network (called U-Net because of its shape)
        # It's like a very complex filter that can process images
        self.model = diffusers.models.UNet2DModel(
            # These numbers determine how powerful the model is
            # More channels (bigger numbers) = more capacity to learn complex patterns
            block_out_channels=(128, 128, 256, 256, 512, 512),
            # These define how the model processes the image at different scales
            # DownBlock2D: looks at increasingly larger patterns in the image
            # AttnDownBlock2D: pays special attention to important parts of the image
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            # UpBlock2D: puts the image back together after analyzing it
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            sample_size=32,  # The size of images we'll work with (32x32 pixels)
        )
        # This is like a recipe for adding and removing noise from images
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            variance_type="fixed_large",  # How much noise to add at each step
            clip_sample=False,  # Don't limit the pixel values
            timestep_spacing="trailing",  # How to space out the denoising steps
        )

    def training_step(self, batch, batch_idx):
        """
        This is what happens in each training step:
        1. Take some real images
        2. Add random noise to them
        3. Ask the model to guess what noise we added
        4. Tell the model how good its guess was
        """
        # Get a batch of real images to train on
        images = batch["images"]
        # Create random noise that looks like our images (same size/shape)
        noise = torch.randn_like(images)
        # Pick random points in the noising process for each image
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        # Add noise to the real images
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # Ask our model to guess what noise we added
        residual = self.model(noisy_images, steps).sample
        # Calculate how wrong the model's guess was (loss)
        loss = torch.nn.functional.mse_loss(residual, noise)
        # Keep track of how well we're doing
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Sets up how the model should learn:
        - AdamW is like a smart teacher that helps the model learn efficiently
        - The scheduler gradually reduces how big the learning steps are
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


class DiffusionData(L.LightningDataModule):
    """
    This class handles all the data preparation.
    It downloads images, processes them, and feeds them to our model.
    """
    def __init__(self):
        super().__init__()
        # Define how to process our images before training
        self.augment = transforms.Compose([
            transforms.Resize(32),  # Make all images 32x32 pixels
            transforms.CenterCrop(32),  # Ensure they're exactly 32x32
            transforms.RandomHorizontalFlip(),  # Randomly flip images to help model learn
            transforms.ToTensor(),  # Convert images to a format our model can use
            transforms.Normalize([0.5], [0.5]),  # Adjust image values to be between -1 and 1
        ])

    def prepare_data(self):
        """Downloads the CIFAR10 dataset (50,000 small color images)"""
        load_dataset("cifar10")

    def train_dataloader(self):
        """
        Creates a system to feed images to our model during training:
        - Loads CIFAR10 dataset
        - Processes images using our defined transformations
        - Creates batches of 128 images
        - Shuffles images so model sees them in different orders
        """
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=4)


# This is where the actual training starts
if __name__ == "__main__":
    model = DiffusionModel()  # Create our model
    data = DiffusionData()    # Prepare our data
    
    # Add early stopping callback
    early_stop_callback = L.callbacks.EarlyStopping(
        monitor='train_loss',
        min_delta=0.00001,  # Minimum change in loss to count as improvement
        patience=10,        # Number of epochs to wait before stopping
        verbose=True,
        mode='min'
    )
    
    trainer = L.Trainer(
        max_epochs=150,
        precision="bf16-mixed",
        callbacks=[early_stop_callback]  # Add the callback here
    )
    trainer.fit(model, data)  # Start training!
