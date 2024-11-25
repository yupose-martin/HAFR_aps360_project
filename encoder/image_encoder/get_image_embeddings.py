import argparse

from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


def encode_image(
    image_path: str, processor: AutoImageProcessor, model: ViTForImageClassification
):
    """
    Get image embeddings using a fine-tuned Vision Transformer model.

    Args:
        image_path (str): Path to the image file.
        processor (AutoImageProcessor): Image processor object.
        model (ViTForImageClassification): Vision Transformer model.

    Returns:
        torch.Tensor: Image embeddings.
    """

    image = Image.open(image_path)

    # Ensure the image is in RGB format
    if image.mode == "RGBA":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    # Use the representation of the CLS token (at index 0) as the image embedding
    embeddings = outputs.hidden_states[-1][:, 0, :]

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
    args = argparser.parse_args()

    # Load the fine-tuned Vision Transformer model
    model = ViTForImageClassification.from_pretrained(args.model_path)
    processor = AutoImageProcessor.from_pretrained(args.model_path)

    # Get image embeddings
    embeddings = encode_image(args.image_path, processor, model)
    print(embeddings)
