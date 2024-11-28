import argparse
import os

import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import (HGTLoader, LinkNeighborLoader,
                                    NeighborLoader)
from transformers import CLIPModel, CLIPProcessor

from GNN.gat import GATModel
from GNN.prepare_graph import prepare_hetero_graph, split_hetero_graph


def train_model(
    base_dir: str,
    base_text_dir: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    log_dir: str,
    batch_size: int,
    num_samples: int,
):
    """
    Train the GATModel with prepared graph data.

    Args:
        base_dir (str): Directory containing recipe images and comments data.
        base_text_dir (str): Directory containing recipe descriptions and steps data.
        processor: Image processor for encoding images.
        model: Vision Transformer model for encoding images.
        log_dir (str): Directory to save training logs.
        batch_size (int): Batch size for training.
        num_samples (int): Number of samples to sample for each type of node.
    """
    # Prepare data
    data = prepare_hetero_graph(base_dir, base_text_dir, log_dir, processor, model)

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
    # train_loader = HGTLoader(
    #     train_data,
    #     num_samples={key: [num_samples] for key in train_data.node_types},
    #     batch_size=batch_size,
    #     input_nodes="user",
    #     shuffle=True,
    # )

    # train_loader = LinkNeighborLoader(
    #     train_data,
    #     num_neighbors=[num_samples],
    #     batch_size=batch_size,
    #     edge_label_index=[
    #         (edge_type, train_data[edge_type].edge_index)
    #         for edge_type in train_data.edge_types
    #     ]
    # )
    train_loader = NeighborLoader(
        train_data,
        num_neighbors={key: [num_samples] for key in train_data.edge_types},
        batch_size=batch_size,
        input_nodes=("item", torch.arange(train_data["item"].num_nodes)),
    )

    val_loader = NeighborLoader(
        val_data,
        num_neighbors={key: [num_samples] * 2 for key in val_data.edge_types},
        batch_size=val_data["item"].num_nodes,
        input_nodes=("item", torch.arange(val_data["item"].num_nodes)),
    )

    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [num_samples] * 2 for key in test_data.edge_types},
        batch_size=test_data["item"].num_nodes,
        input_nodes=("item", torch.arange(test_data["item"].num_nodes)),
    )
    # train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader([val_data], batch_size=batch_size, shuffle=False)

    # Initialize model
    gat_model = GATModel(
        unique_users=train_data["user"].x.size(0),
        item_dim=train_data["item"].x.size(1),
        ingredient_dim=train_data["ingredient"].x.size(1),
        hidden_dim=10,
        num_layers=1,
        num_heads_per_layer=1,
        lr=1e-4,
        use_weighted_loss=False,
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
        "base_dir",
        type=str,
        help="Path to the directory containing recipe image and comments data",
    )
    argparser.add_argument(
        "base_text_dir",
        type=str,
        help="Path to the directory containing recipe text data",
    )
    argparser.add_argument(
        "model_path", type=str, help="Path to the fine-tuned CLIP model"
    )
    argparser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Path to the directory to save training logs",
    )
    args = argparser.parse_args()

    # Load the CLIP model
    print("Loading CLIP model...")
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

    # Train the GAT model
    train_model(
        args.base_dir,
        args.base_text_dir,
        clip_processor,
        clip_model,
        args.log_dir,
        batch_size=16,
        num_samples=20,
    )
