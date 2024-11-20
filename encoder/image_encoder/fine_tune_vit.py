import argparse
import os

import numpy as np
import torch
from encoder.image_encoder.prepare_dataset import (RecipeDataset,
                                                   load_recipe_data)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (Trainer, TrainerCallback, TrainingArguments,
                          ViTForImageClassification, ViTImageProcessorFast)


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs, output_dir, processor=None):
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir
        self.processor = processor

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every_n_epochs == 0:
            save_dir = os.path.join(
                self.output_dir, f"checkpoint-epoch-{int(state.epoch)}"
            )
            kwargs["model"].save_pretrained(save_dir)
            print(f"Model saved to {save_dir}")

            if self.processor is not None:
                self.processor.save_pretrained(save_dir)
                print(f"Processor saved to {save_dir}")


def compute_metrics(eval_pred):
    """
    Compute accuracy for evaluation.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.

    Returns:
        Dict[str, float]: A dictionary with accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predicted labels
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Vision Transformer model")
    parser.add_argument(
        "base_directory",
        type=str,
        help="Path to the directory containing recipe subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to the output directory",
    )
    args = parser.parse_args()

    recipes = load_recipe_data(args.base_directory)

    # Set device
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
    print(f"Using device: {device}")

    # Map recipe names to unique labels and prepare data
    label_to_index = {recipe["recipe_name"]: idx for idx, recipe in enumerate(recipes)}
    index_to_label = {idx: recipe for recipe, idx in label_to_index.items()}
    image_paths = [img for recipe in recipes for img in recipe["image_paths"]]
    labels = [
        label_to_index[recipe["recipe_name"]]
        for recipe in recipes
        for img in recipe["image_paths"]
    ]

    # Split the data into training and validation sets
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Prepare datasets for training and validation
    processor = ViTImageProcessorFast.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    train_dataset = RecipeDataset(train_image_paths, train_labels, processor)
    val_dataset = RecipeDataset(val_image_paths, val_labels, processor)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Load a pre-trained ViT model and configure it for classification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(label_to_index),
        id2label=index_to_label,
        label2id=label_to_index,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Output directory
        evaluation_strategy="epoch",  # Evaluate the model after every epoch
        learning_rate=5e-5,  # Learning rate for the optimizer
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        save_strategy="no",  # Use a callback to save the model
        num_train_epochs=10,  # Number of training epochs
        weight_decay=0.01,  # Weight decay for regularization
        logging_dir="./logs",  # Directory for storing logs
        logging_strategy="epoch",  # Log metrics after every epoch
        push_to_hub=False,  # Do not push the model to the Hugging Face Hub
    )

    save_callback = SaveEveryNEpochsCallback(
        save_every_n_epochs=1, output_dir=args.output_dir, processor=processor
    )

    # Initialize the Trainer with model, arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[save_callback],
    )

    # Fine-tune the ViT model
    print("Fine-tuning the model...")
    trainer.train()

    # Evaluate on the validation dataset
    results = trainer.evaluate()
    print(f"Validation Results: {results}")
