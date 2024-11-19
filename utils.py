# utils.py
import torchvision
from torch.utils.data import DataLoader
import torch
from dataset import TrainDataset

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves the model's state to a checkpoint file."""
    print("=> Saving checkpoint")
    torch.save(state, filename)  # Save the model state to a file

def load_checkpoint(checkpoint, model):
    """Loads a checkpoint into the model."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])  # Load model weights from checkpoint

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Returns DataLoader for training and validation datasets.

    Args:
    - train_dir (str): Path to the training images.
    - train_maskdir (str): Path to the training masks.
    - val_dir (str): Path to the validation images.
    - val_maskdir (str): Path to the validation masks.
    - batch_size (int): Batch size for DataLoader.
    - train_transform (callable): Transformations to apply to the training images.
    - val_transform (callable): Transformations to apply to the validation images.
    - num_workers (int): Number of workers for DataLoader.
    - pin_memory (bool): Whether to pin memory for faster data loading.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """
    # Initialize training and validation datasets
    train_ds = TrainDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    # Create DataLoader for the training dataset
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # Initialize validation dataset
    val_ds = TrainDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    # Create DataLoader for the validation dataset
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader  # Return both train and val loaders

def check_accuracy(loader, model, device="cuda"):
    """
    Checks the accuracy of the model on the given data loader.

    Args:
    - loader (DataLoader): The DataLoader containing the dataset to check.
    - model (torch.nn.Module): The trained model.
    - device (str): Device to run the model on (default is 'cuda').

    Prints accuracy and dice score.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:
            x = x.to(device)  # Move input image to the specified device
            y = y.to(device).unsqueeze(1)  # Move mask to device, and add channel dimension
            preds = torch.sigmoid(model(x))  # Get model predictions with sigmoid activation
            preds = (preds > 0.5).float()  # Apply threshold to get binary predictions
            num_correct += (preds == y).sum()  # Count the number of correct pixels
            num_pixels += torch.numel(preds)  # Count total pixels
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)  # Compute Dice score
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Saves model predictions as images.

    Args:
    - loader (DataLoader): DataLoader for the dataset.
    - model (torch.nn.Module): The trained model.
    - folder (str): Directory to save the images (default is 'saved_images/').
    - device (str): Device to run the model on (default is 'cuda').
    """
    model.eval()  # Set the model to evaluation mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)  # Move input image to device
        with torch.no_grad():  # Disable gradient computation
            preds = torch.sigmoid(model(x))  # Get model predictions
            preds = (preds > 0.5).float()  # Apply threshold to get binary predictions
        # Save predictions and true masks as images
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train()  # Switch the model back to training mode
