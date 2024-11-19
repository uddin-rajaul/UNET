# dataset.py

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
  """
  Dataset for training the UNET model.

  This dataset loads image and mask pairs from the specified directories.
  It applies transformations to the data during loading.
  """

  def __init__(self, image_dir, mask_dir, transform=None):
    """
    Initializes the dataset with the directories for images and masks.

    Parameters:
    - image_dir (str): Path to the directory containing the images.
    - mask_dir (str): Path to the directory containing the corresponding masks.
    - transform (callable, optional): A function/transform to apply to the images and masks.
    """
    self.image_dir = image_dir           # Store the path to the image directory
    self.mask_dir = mask_dir             # Store the path to the mask directory
    self.transform = transform           # Store any transformations to apply
    self.images = os.listdir(image_dir)  # List all image filenames in the image directory

  def __len__(self):
    """
    Returns the number of images in the dataset.

    Returns:
    - int: The total number of images in the dataset.
    """
    return len(self.images)

  def __getitem__(self, index):
    """
    Fetches the image and its corresponding mask at the specified index.

    Parameters:
    - index (int): Index to fetch a specific image-mask pair.

    Returns:
    - image (numpy.ndarray): The image at the specified index.
    - mask (numpy.ndarray): The corresponding mask for the image.
    """

    # Get the file paths for the image and its corresponding mask
    img_path = os.path.join(self.image_dir, self.images[index])  # Construct full image path
    mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))  # Construct full mask path

    # Open the image and mask, converting them to numpy arrays
    image = np.array(Image.open(img_path).convert("RGB"))  # Convert image to RGB and then to a numpy array
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # Convert mask to grayscale and to numpy array

    # Convert mask pixel values: 255 becomes 1, others remain 0 (binary mask)
    mask[mask == 255] = 1.0

    # Apply transformations if provided
    if self.transform is not None:
      augmentations = self.transform(image=image, mask=mask)  # Apply the transformations
      image = augmentations['image']  # Get the transformed image
      mask = augmentations['mask']    # Get the transformed mask

    return image, mask  # Return the image and mask as a tuple
