import json
import os
import random

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor, ViTImageProcessor


def load_recipe_data(base_dir: str) -> list[dict[str, any]]:
    """
    Load recipe data from a given directory.

    Args:
        base_dir (str): The base directory containing recipe subdirectories.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with keys `recipe_name`,
                              `image_paths`, and `comments`.
    """
    recipes = []
    for recipe_name in os.listdir(base_dir):
        recipe_path = os.path.join(base_dir, recipe_name)
        if os.path.isdir(recipe_path):
            images_dir = os.path.join(recipe_path, "images")
            comments_file = os.path.join(recipe_path, "comments", "comments.json")

            # Load image paths
            image_paths = [
                os.path.join(images_dir, fname)
                for fname in os.listdir(images_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            # Load comments
            comments = []
            if os.path.exists(comments_file):
                with open(comments_file, "r", encoding="utf-8") as f:
                    comments = json.load(f)

            recipes.append(
                {
                    "recipe_name": recipe_name,
                    "image_paths": image_paths,
                    "comments": comments,
                }
            )
    return recipes


class RecipeDataset(Dataset):
    def __init__(
        self, image_paths: list[str], labels: list[int], processor: AutoImageProcessor
    ):
        """
        Dataset for recipes with image paths and labels.

        Args:
            image_paths (List[str]): List of paths to image files.
            labels (List[int]): List of labels corresponding to each image.
            processor (AutoImageProcessor): Processor for image preprocessing.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, any]:
        """
        Get an item by index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Dict[str, Any]: Preprocessed image tensor and corresponding label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            processed = self.processor(images=img, return_tensors="pt")

        return {"pixel_values": processed["pixel_values"].squeeze(0), "label": label}


def show_random_image_with_label(
    dataset: VisionDataset, index_to_label: dict[int, str]
) -> None:
    """
    Displays a random image from the given dataset along with its corresponding label.

    Args:
        dataset (VisionDataset): A dataset containing images and their labels.
            It should support indexing to retrieve individual samples as (image, label) pairs.
        index_to_label (Dict[int, str]): A dictionary mapping label indices (int) to human-readable label names (str).

    Returns:
        None
    """
    # Generate a random index
    random_index = random.randint(0, len(dataset) - 1)

    # Retrieve the image and label
    data_point = dataset[random_index]
    image_tensor, label = data_point["pixel_values"], data_point["label"]

    # Get normalization parameters from the processor
    mean = torch.tensor(dataset.processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(dataset.processor.image_std).view(-1, 1, 1)

    # Undo normalization
    image_tensor = image_tensor * std + mean

    # Clamp pixel values to [0, 1]
    image_tensor = image_tensor.clamp(0, 1)

    # Convert tensor to PIL image
    image = ToPILImage()(image_tensor)

    # Get the label name
    label_name = index_to_label.get(label, "Unknown Label")

    # Display the image and label
    plt.imshow(image)
    plt.title(f"Label: {label_name}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    print("Preparing dataset...")
    # Load recipe data
    base_dir = "/Users/deyucao/HAFR_aps360_project/data"
    recipes = load_recipe_data(base_dir)

    print(f"Loaded {len(recipes)} recipes")

    # Define indices for labels
    index_to_label = {i: recipe["recipe_name"] for i, recipe in enumerate(recipes)}
    label_to_index = {v: k for k, v in index_to_label.items()}

    # Create labels
    image_paths = [img for recipe in recipes for img in recipe["image_paths"]]
    labels = [
        label_to_index[recipe["recipe_name"]]
        for recipe in recipes
        for img in recipe["image_paths"]
    ]

    # Load image processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Create dataset
    dataset = RecipeDataset(
        image_paths=[img for recipe in recipes for img in recipe["image_paths"]],
        labels=labels,
        processor=processor,
    )

    # Display a random image with its label
    show_random_image_with_label(dataset, index_to_label)
