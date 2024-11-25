import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import HGTLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from GNN.gat import GATModel
from GNN.prepare_graph import prepare_hetero_graph, split_hetero_graph


class ValidationCallback(Callback):
    def __init__(self, val_loader):
        self.val_loader = val_loader

    def on_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        print("Validating...")
        total_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(pl_module.device)
                loss, accuracy = pl_module.validation_step(batch)
                total_loss += loss.item()
                total_accuracy += accuracy.item()
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        print(f"Validation Loss: {avg_loss}, Accuracy: {avg_accuracy}")
        pl_module.train()


def train_model(
    base_dir: str,
    processor: AutoImageProcessor,
    model: ViTForImageClassification,
    log_dir: str,
    batch_size: int,
    num_neighbors: int,
):
    """
    Train the GATModel with prepared graph data.

    Args:
        base_dir (str): Directory containing recipe data.
        processor: Image processor for encoding images.
        model: Vision Transformer model for encoding images.
        log_dir (str): Directory to save training logs.
        batch_size (int): Batch size for training.
        num_neighbors (int): Number of neighbors to sample for each node.
    """
    # Prepare data
    data = prepare_hetero_graph(base_dir, processor, model)

    # Split data
    train_data, val_data, test_data = split_hetero_graph(data)

    print(f"Users in train data: {train_data['user'].num_nodes}")
    print(f"Items in train data: {train_data['item'].num_nodes}")
    print(
        f"Interactions in train data: {train_data['user', 'rates', 'item'].num_edges}"
    )

    print(f"Users in val data: {val_data['user'].num_nodes}")
    print(f"Items in val data: {val_data['item'].num_nodes}")
    print(f"Interactions in val data: {val_data['user', 'rates', 'item'].num_edges}")

    # Create data loaders
    train_loader = HGTLoader(
        train_data,
        num_samples={
            ("user", "rates", "item"): num_neighbors,
            ("item", "rated_by", "user"): num_neighbors,
        },
        batch_size=batch_size,
        input_nodes="user",
    )

    val_loader = HGTLoader(
        val_data,
        num_samples={
            ("user", "rates", "item"): num_neighbors,
            ("item", "rated_by", "user"): num_neighbors,
        },
        batch_size=batch_size,
        input_nodes="user",
    )

    # Initialize model
    gat_model = GATModel(
        user_dim=train_data["user"].x.size(1),
        item_dim=train_data["item"].x.size(1),
        rating_dim=6,  # Assuming ratings are in the range 1-5
        hidden_dim=128,
        output_dim=128,
        num_layers=1,
        lr=0.001,
    )

    # Training setup
    logger = CSVLogger(log_dir, name="GAT_training_logs")
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        devices=1,
        accelerator="cpu",
        callbacks=[ValidationCallback(val_loader)],
    )

    # Train the model
    print("Training GAT model...")
    trainer.fit(gat_model, train_loader)


if __name__ == "__main__":
    print("start")
    argparser = argparse.ArgumentParser(
        description="Train a GAT model for recipe recommendations"
    )

    argparser.add_argument(
        "base_dir", type=str, help="Path to the directory containing recipe data"
    )
    argparser.add_argument(
        "model_path", type=str, help="Path to the fine-tuned Vision Transformer model"
    )
    argparser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Path to the directory to save training logs",
    )
    args = argparser.parse_args()

    # Load the fine-tuned Vision Transformer model
    print("Loading Vision Transformer model...")
    model = ViTForImageClassification.from_pretrained(args.model_path)
    processor = AutoImageProcessor.from_pretrained(args.model_path)

    # Train the GAT model
    print("Training GAT model...")
    train_model(args.base_dir, processor, model, "./logs", 1, 10)
