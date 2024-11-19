# train.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 2  # Number of CPU cores to use for data loading
IMAGE_HEIGHT = 160  # Resize the image to this height
IMAGE_WIDTH = 240   # Resize the image to this width
PIN_MEMORY = True  # If True, DataLoader will use pinned memory
LOAD_MODEL = False  # Whether to load a pre-trained model
TRAIN_IMG_DIR = "data/train_images/"  # Path to training images
TRAIN_MASK_DIR = "data/train_masks/"  # Path to training masks
VAL_IMG_DIR = "data/val_images/"  # Path to validation images
VAL_MASK_DIR = "data/val_masks/"  # Path to validation masks


def train(loader, model, optimizer, loss_fn, scaler):
    """
    Trains the model for one epoch using mixed precision.

    Args:
    - loader (DataLoader): The data loader for the training set.
    - model (nn.Module): The model to train.
    - optimizer (optim.Optimizer): The optimizer for updating the model's weights.
    - loss_fn (nn.Module): The loss function to minimize.
    - scaler (torch.cuda.amp.GradScaler): Scales the loss for mixed precision training.
    """
    loop = tqdm(loader)  # Progress bar for training loop

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)  # Move data to the specified device
        targets = targets.float().unsqueeze(1).to(DEVICE)  # Move targets (masks) to the device

        # Forward pass with automatic mixed precision (AMP)
        with torch.cuda.amp.autocast():  # Enables mixed precision for faster training
            predictions = model(data)  # Get model predictions
            loss = loss_fn(predictions, targets)  # Compute the loss

        # Backward pass and optimization step
        optimizer.zero_grad()  # Clear gradients from the previous step
        scaler.scale(loss).backward()  # Scale and compute gradients
        scaler.step(optimizer)  # Update model weights
        scaler.update()  # Update the scaler for the next iteration

        # Update the progress bar with the loss value
        loop.set_postfix(loss=loss.item())


def main():
    # Define data augmentation and transformation for training and validation sets
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),  # Resize images
            A.Rotate(limit=35, p=1.0),  # Random rotation
            A.HorizontalFlip(p=0.1),  # Horizontal flip with 10% probability
            A.VerticalFlip(p=0.1),  # Vertical flip with 10% probability
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),  # Normalize image pixels
            ToTensorV2(),  # Convert to PyTorch tensor
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),  # Resize for validation set
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Initialize model, loss function, and optimizer
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # UNET model for segmentation
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss with logits (for binary segmentation)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

    # Get the training and validation data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # Optionally load a pre-trained model
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()  # Initialize mixed precision scaler
    best_loss = float('inf')  # Track the best validation loss for model saving

    # Training loop for each epoch
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)  # Train the model for one epoch

        # Check accuracy and calculate validation loss
        val_loss = check_accuracy(val_loader, model, device=DEVICE)

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, model_save_path)  # Save the model
            print(f"Best model saved to {model_save_path} with loss {best_loss}")

        # Save example predictions for visualization
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()  # Run the main function
