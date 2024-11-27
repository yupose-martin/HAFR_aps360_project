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


def load_recipe_data(base_dir: str , base_text_dir: str) -> list[dict[str, any]]:
    """
    Load recipe data from a given directory.

    Args:
        base_dir (str): The base directory containing recipe subdirectories.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with keys `recipe_name`,
                              `image_paths`, and `comments`.
    """
    image_base_path = base_dir
    text_base_path = base_text_dir
    image_folder_names = [f for f in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, f))]
    text_folder_names = [f for f in os.listdir(text_base_path) if os.path.isdir(os.path.join(text_base_path, f))]
    # common foler names
    common_folder_names = list(set(image_folder_names).intersection(text_folder_names))

    print(len(common_folder_names))
    
    recipes = []
    #load the paths of the images and the text
    for folder_name in common_folder_names:
        image_folder_path = os.path.join(image_base_path, folder_name)
        text_folder_path = os.path.join(text_base_path, folder_name)
        
        
        # there's one text description for each foler. But there's multiple images in each folder
        # so we need to create a pair of image and text for each image in the folder
        # and we create dataset based on that
        # then we can create a dataloader for the dataset
        # then we can load the model and fine-tune the model based on our dataloader
        image_path = os.path.join(image_folder_path, image_folder_path)
        
        image_path = os.path.join(image_folder_path, "images")
        comments_file_path = os.path.join(image_folder_path, "comments", "comments.json")
        # there's many images in the image folder
        
        # there's one text description for each foler. We need to load the description and steps and add them together
        text_path_descriptions = os.path.join(text_folder_path, "descriptions")
        text_path_descriptions = os.path.join(text_path_descriptions, "descriptions.txt")
        text_path_steps = os.path.join(text_folder_path, "steps")
        text_path_steps = os.path.join(text_path_steps, "steps.txt")
        
        # load the text description and steps and add them up together to form a string
        with open(text_path_descriptions, "r", encoding="utf-8") as f:
            description = f.read()
        with open(text_path_steps, "r", encoding="utf-8") as f:
            steps = f.read()
        text = description + steps
        with open(comments_file_path, "r", encoding="utf-8") as f:
            comments = f.read()
        
        for image_file_name in os.listdir(image_path):
            image_file_path = os.path.join(image_path, image_file_name)
            
            recipes.append(
                {
                    "recipe_name": folder_name,
                    "image_paths": image_file_path,
                    "comments": comments,
                    "description": text,
                }
            )
    return recipes

# Example on how to use this function
# recipe = load_recipe_data(base_dir=".\\data_image\\",base_text_dir=".\\data_text\\")

# print(f"len(recipe): {len(recipe)}")

# print(f"recipe[0]['comments']: {recipe[0]['comments']}")
# print(f"recipe[0]['description']: {recipe[0]['description']}")
# print(f"recipe[0]['image_paths']: {recipe[0]['image_paths']}")
# print(f"recipe[0]['recipe_name']: {recipe[0]['recipe_name']}")

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
    base_text_dir = "your base dir for text"
    recipes = load_recipe_data(base_dir, base_text_dir)

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
