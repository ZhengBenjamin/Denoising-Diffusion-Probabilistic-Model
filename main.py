import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
import torchvision.io as io

from process import DiffusionForward 
from reverse import DiffusionReverse
from model import *
from PIL import Image
import math




# Convert images to tensor: 
def load_images(image_dir, num_images=10, image_size=(64, 64), device=None):
    """
    Loads and preprocesses a fixed number of images from a directory.
    Returns tensor (num_images, 3, H, W).
    """

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), 
    ])
    
    images = []
    for i in range(num_images):
        image_path = os.path.join(image_dir, f"{i}.png")
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)
    
    # Stack images into a batch tensor and move to device
    x0 = torch.stack(images, dim=0)
    if device is not None:
        x0 = x0.to(device)
    return x0

device = torch.device("cuda")

# Forward Process
def demo_forward_process(num_images=2, num_steps=1000, image_size=(64, 64)):
    forward = DiffusionForward(num_steps=num_steps)
    x0 = load_images("data/simp", num_images=num_images, image_size=image_size, device=device)

    sampled_images = []

    # 10 intermediate samples 
    for i in range(0, num_steps, num_steps//10):
        sampled_images.append(forward.q_sample(x0, i))

    fig, axs = plt.subplots(num_images, 10, figsize=(10,5))
    for i in range(num_images):
        for j in range(len(sampled_images)):
            axs[i][j].imshow(sampled_images[j][i].cpu().permute(1, 2, 0).numpy().clip(0,1))
            axs[i][j].axis('off')
    plt.tight_layout()
    plt.show()

def train(num_images=50, num_steps=1000, image_size=(64,64), epochs=1000, lr=1e-4, batch_size=8, continue_training=False):
    model_path = "model.pth"
    torch.backends.cudnn.benchmark = True # yeah idk for some reason this is faster

    # Initialize a new model and optimizer (and scheduler) always
    model = UNet(in_channels=3, out_channels=3, base_channels=512, time_emb_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    start_epoch = 0

    # If continuing training, load checkpoint state
    if continue_training and os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        # Load saved states if available
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Continuing training from epoch {start_epoch} using checkpoint at {model_path}")
        else:
            # Fallback if only model was saved previously.
            model = checkpoint
            print("Checkpoint does not contain training state. Restarting epochs from 0.")
        # model=checkpoint
    else:
        print("Training a new model")

    image_paths = [os.path.join("data/simp", f"{i}.png") for i in range(1, num_images + 1)]
    batch_load_size = 10000
    all_images = []

    for b in range(0, num_images, batch_load_size):
        batch_paths = image_paths[b:b+batch_load_size]
        batch_images = []
        for i, path in enumerate(batch_paths, start=b):
            # Read image: returns tensor [C, H, W] in uint8
            img_tensor = io.read_image(path).float() / 255.0  # scale to [0,1]
            # Move to GPU before resizing
            img_tensor = img_tensor
            # Add batch dimension and resize on GPU using bilinear interpolation
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=image_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            batch_images.append(img_tensor)
            if ((i - b) % 1000) == 0:
                print(f"Loaded {i+1}/{num_images} images")
        all_images.extend(batch_images)
        print(f"Completed batch {(b // batch_load_size) + 1}")

    # Stack into a single Tensor [num_images, 3, H, W] on GPU
    dataset = torch.stack(all_images, dim=0).to(device)

    for epoch in range(start_epoch, epochs):
        model.train()

        # Select a random batch from our in-memory dataset
        idx = np.random.choice(num_images, batch_size, replace=False)
        x0 = dataset[idx]  # shape: [batch_size, 3, H, W]

        # Draw random diffusion steps for each sample in the batch
        t = torch.randint(0, num_steps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        x_t = DiffusionForward(num_steps=num_steps).q_sample(x0, t, noise=noise)

        # Forward pass through the model
        predicted_noise = model(x_t, t)

        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backprop & optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item():.6f}")

        if (epoch + 1) % 1000 == 0:
            # Save checkpoint as a dictionary containing training state
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }
            torch.save(checkpoint, model_path)
            print(f"Model saved at epoch {epoch + 1}")
            gen_images(model, num_images=3, num_steps=num_steps, shape=image_size, show=False, epochs=epoch + 1)

    # Save the final model checkpoint
    checkpoint = {
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    torch.save(checkpoint, model_path)
    return x0, model


# Reverse Process
def demo_reverse_process(model, num_images=5, num_steps=1000, shape=(64, 64), show=True):
    reverse = DiffusionReverse(model=model, num_steps=num_steps, device=device)
    
    with torch.no_grad():
        # Start from pure Gaussian noise for all images at once
        current = torch.randn(num_images, 3, shape[0], shape[1], device=device)
        snapshots = [[] for _ in range(num_images)]

        for step in range(10):
            start_t = num_steps - step * (num_steps // 10) - 1
            end_t = start_t - (num_steps // 10)

            # Reverse process from t = start_t down to end_t
            for t in range(start_t, end_t, -1):
                current = reverse.p_sample(current, t)

            # Collect snapshots for all images at this step
            for i in range(num_images):
                snapshots[i].append(current[i].detach().cpu())

        fig, axs = plt.subplots(num_images, 10, figsize=(15, 15))
        for example in range(num_images):
            for idx, img in enumerate(snapshots[example]):
                img_np = img.permute(1, 2, 0).numpy().clip(0, 1)
                axs[example, idx].imshow(img_np)
                axs[example, idx].axis("off")

            print(f"Example {example + 1}: Final generated image shape: {current[example].shape}")

        plt.tight_layout()

        if show:
            plt.show()

        plt.savefig("reverse_process_output.jpg")

def gen_images(model, num_images=5, num_steps=1000, shape=(64, 64), show=True, epochs=0):
    reverse = DiffusionReverse(model=model, num_steps=num_steps, device=device)
    generated_images = []
    batch_size = 128

    num_batches = math.ceil(num_images / batch_size)

    print(f"Generating {num_images} images in batches of {batch_size}...")

    with torch.no_grad():
        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_images - len(generated_images))
            batch = reverse.sample((current_batch_size, 3, shape[0], shape[1]))
            for i in range(current_batch_size):
                image = batch[i].detach().cpu()
                generated_images.append(image)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Create a dynamic grid based on number of images
    grid_size = int(math.ceil(math.sqrt(num_images)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axs = axs.flatten()
    for i in range(len(generated_images)):
        img = generated_images[i].squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        axs[i].imshow(img)
        axs[i].axis("off")
    for j in range(len(generated_images), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(f"Generated_Images_{epochs}.jpg")

# demo_forward_process(num_images=200, num_steps=750, image_size=(64, 64))
# x0, model = train(num_images=9800, num_steps=500, image_size=(64,64), epochs=10000000, lr=1e-4, batch_size=32, continue_training=True)

# model = torch.load("dog32.pth", weights_only=False) # 32x32 t=1000 base=256 time_emb=512 -> dog32 model used old saving format, comment out if instance.... block 

# checkpoint = torch.load("dog64.pth", weights_only=False) # 64x64 t=750 base=256 time_emb=512
checkpoint = torch.load("human64.pth", weights_only=False) # 64x64 t=750 base=384 time_emb=512
# checkpoint = torch.load("simp64.pth", weights_only=False) # 64x64 t=750 base=256 time_emb=384

# checkpoint = torch.load("model.pth", weights_only=False) # Default for current model
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state = checkpoint["model_state_dict"]
    model = UNet(in_channels=3, out_channels=3, base_channels=384, time_emb_dim=512).to(device)
    model.load_state_dict(model_state)
else:
    model = checkpoint
demo_reverse_process(model, num_images=10, num_steps=750, shape=(64, 64), show=False)
gen_images(model, num_images=100, num_steps=750, shape=(64, 64), show=False)