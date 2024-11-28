import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (CLIPImageProcessor, CLIPModel, CLIPProcessor,
                          CLIPTokenizerFast)


def encode_images(
    image_paths: list[str],
    processor: CLIPImageProcessor,
    model: CLIPModel,
    device="cuda",
):
    """
    Get image embeddings using a fine-tuned CLIPModel

    Args:
        image_path (str): Path to the image file to encode.
        processor (CLIPProcessor): CLIPProcessor.
        model (CLIPModel): CLIPModel

    Returns:
        torch.Tensor: Image embeddings.
    """
    model = model.to(device)

    images = []
    for image_path in image_paths:
        image = Image.open(image_path)

        # Ensure the image is in RGB format
        if image.mode == "RGBA":
            image = image.convert("RGB")
        images.append(image)

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    return embeddings


def encode_texts(
    texts: list[str], processor: CLIPTokenizerFast, model: CLIPModel, device="cuda"
):
    """
    Get text embeddings using a fine-tuned CLIPModel

    Args:
        texts (list[str]): List of texts to encode.
        processor (CLIPProcessor): CLIPProcessor.
        model (CLIPModel): CLIPModel

    Returns:
        torch.Tensor: Image embeddings.
    """
    model = model.to(device)

    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)

    return embeddings


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Get image embeddings using a fine-tuned Vision Transformer model"
    )
    argparser.add_argument(
        "image_path", type=str, help="Path to the image file to encode"
    )
    argparser.add_argument(
        "model_path", type=str, help="Path to the fine-tuned Vision Transformer model"
    )
    argparser.add_argument("description", type=str, help="Description information")

    args = argparser.parse_args()

    print(f"Image path: {args.image_path}")
    print(f"Text description: {args.description}")

    # get text embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )  # Assume the model is fine-tuned on this model

    # later change to our fine-tuned clip model
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        device_map=device,
        torch_dtype=torch.float32,
    )

    clip_model.load_state_dict(torch.load(args.model_path))

    text_embeddings = encode_texts(
        texts=args.description,
        processor=clip_processor,
        model=clip_model,
        device=device,
    )

    # # Get image embeddings
    image_embeddings = encode_images(
        image_paths=[args.image_path],
        processor=clip_processor,
        model=clip_model,
        device=device,
    )

    # Compute similarity between text and image embeddings
    similarity = F.cosine_similarity(text_embeddings, image_embeddings).item()

    print(f"Similarity between text and image embeddings: {similarity:.4f}")
