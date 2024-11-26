import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import HGTLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from GNN.gat import GATModel
from GNN.prepare_graph import prepare_hetero_graph, split_hetero_graph


def train_model(
    base_dir: str,
    processor: AutoImageProcessor,
    model: ViTForImageClassification,
    log_dir: str,
    batch_size: int,
    num_samples: int,
):
    """
    Train the GATModel with prepared graph data.

    Args:
        base_dir (str): Directory containing recipe data.
        processor: Image processor for encoding images.
        model: Vision Transformer model for encoding images.
        log_dir (str): Directory to save training logs.
        batch_size (int): Batch size for training.
        num_samples (int): Number of samples to sample for each type of node.
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
        num_samples={key: [num_samples] for key in train_data.node_types},
        batch_size=batch_size,
        input_nodes="user",
        shuffle=True,
    )

    val_loader = HGTLoader(
        val_data,
        num_samples={key: [num_samples] for key in val_data.node_types},
        batch_size=val_data["user"].num_nodes,
        input_nodes="user",
        shuffle=False,
    )

    test_loader = HGTLoader(
        test_data,
        num_samples={key: [num_samples] for key in test_data.node_types},
        batch_size=test_data["user"].num_nodes,
        input_nodes="user",
        shuffle=False,
    )
    # train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader([val_data], batch_size=batch_size, shuffle=False)

    # Initialize model
    gat_model = GATModel(
        unique_users=train_data["user"].x.size(0),
        item_dim=train_data["item"].x.size(1),
        rating_dim=6,  # Assuming ratings are in the range 1-5
        hidden_dim=4,
        output_dim=32,
        num_layers=1,
        num_heads_per_layer=1,
        lr=1e-4,
        use_weighted_loss=True,
    )

    # Training setup
    logger = CSVLogger(log_dir, name="GAT_training_logs")
    trainer = Trainer(
        max_epochs=30,
        logger=logger,
        devices=1,
        accelerator="gpu",
    )

    # Train the model
    print("Training GAT model...")
    trainer.fit(gat_model, train_loader, val_loader)
    trainer.test(gat_model, test_loader)


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
    train_model(args.base_dir, processor, model, "./logs", 100, 100)
