import argparse

from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import clip
from transformers import CLIPProcessor, CLIPModel
import requests
from transformers import CLIPModel
import torch

def image_encoder(
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
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    # Use the representation of the CLS token (at index 0) as the image embedding
    embeddings = outputs.hidden_states[-1][:, 0, :]

    return embeddings

def text_encoder(
    text: str, image_path: str, processor: CLIPProcessor, model: CLIPModel
):
    """
    Get text embeddings using a fine-tuned CLIPModel (image is also required????!!)

    Args:
        text (str): text
        image_path (str): image path
        processor (CLIPProcessor): CLIPProcessor.
        model (CLIPModel): CLIPModel

    Returns:
        torch.Tensor: Image embeddings.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    inputs = processor(text=[text], images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    
    # modify based on what embedding you want
    embeddings = outputs.logits_per_text

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
    argparser.add_argument(
        "description", type=str, help="Description information"
    )
    
    args = argparser.parse_args()

    # Load the fine-tuned Vision Transformer model
    model = ViTForImageClassification.from_pretrained(args.model_path)
    processor = AutoImageProcessor.from_pretrained(args.model_path)

    # Get image embeddings
    embeddings = image_encoder(args.image_path, processor, model)
    print(embeddings)
    
    # # get text embeddings
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device}")
    
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # #later change to our fine-tuned clip model
    # clip_model = CLIPModel.from_pretrained(
    # "openai/clip-vit-base-patch32",
    # device_map=device,
    # torch_dtype=torch.float32,
    # )
    
    # text_embeddings = text_encoder(text=args.description, image_path=args.image_path,
    #                                processor=clip_processor, model=clip_model)
    # print(text_embeddings)
