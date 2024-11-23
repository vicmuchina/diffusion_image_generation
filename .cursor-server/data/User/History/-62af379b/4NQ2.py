import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from train import DiffusionModel


def main(
    checkpoint: Path = Path("lightning_logs/version_0/checkpoints/epoch=99-step=9800.ckpt"),
    num_timesteps: int = 1000,
    num_samples: int = 64,
    seed: int = 0,
):
    """
    Generates new images using a trained diffusion model.
    
    Args:
        checkpoint (Path): Path to the trained model checkpoint file.
        num_timesteps (int): Number of denoising steps (higher means better quality but slower).
        num_samples (int): Number of images to generate.
        seed (int): Seed for random number generation to make results reproducible.
    
    Process:
        1. Sets the random seed for reproducibility.
        2. Determines whether to use GPU or CPU.
        3. Loads the trained diffusion model from the checkpoint.
        4. Sets up the diffusion scheduler and pipeline.
        5. Generates images by progressively denoising from random noise.
        6. Arranges generated images into a grid and saves them as a PNG file.
    """
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with device:
        model = DiffusionModel()
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(variance_type="fixed_large", timestep_spacing="trailing")
    pipe = diffusers.DDPMPipeline(model.model, scheduler)
    pipe = pipe.to(device=device)
    
    with torch.inference_mode():
        (pil_images, ) = pipe(
            batch_size=num_samples, 
            num_inference_steps=num_timesteps, 
            output_type="pil", 
            return_dict=False
        )
    images = torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_images])
    image_grid = tv.utils.make_grid(images, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "generated.png"
    tv.utils.save_image(image_grid, filename)
    print(f"Generated images saved to {filename}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
