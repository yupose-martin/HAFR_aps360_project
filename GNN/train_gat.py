import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import HGTLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from GNN.gat import GATModel
from GNN.prepare_graph import prepare_hetero_graph, split_hetero_graph

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    """
    Train the GATModel with configurations provided via Hydra.
    """
    print(cfg)

    # Extract training configurations
    base_dir = cfg.training.base_dir
    log_dir = cfg.training.log_dir
    model_save_path = cfg.training.model_save_path

    # Extract hyperparameters
    batch_size = cfg.hyperparameters.batch_size
    num_samples = cfg.hyperparameters.num_samples
    hidden_dim = cfg.hyperparameters.hidden_dim
    output_dim = cfg.hyperparameters.output_dim
    lr = cfg.hyperparameters.lr
    num_layers = cfg.hyperparameters.num_layers
    num_heads_per_layer = cfg.hyperparameters.num_heads_per_layer
    use_weighted_loss = cfg.hyperparameters.use_weighted_loss
    max_epochs = cfg.hyperparameters.max_epochs

    # Load the Vision Transformer model and processor
    print("Loading Vision Transformer model...")
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    # Prepare graph data
    data = prepare_hetero_graph(cfg, processor, model)
    train_data, val_data, test_data = split_hetero_graph(data)

    # Print data stats for debugging
    print(f"Users in train data: {train_data['user'].num_nodes}")
    print(f"Items in train data: {train_data['item'].num_nodes}")
    print(
        f"Interactions in train data: {train_data['user', 'rates', 'item'].num_edges}"
    )

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

    # Initialize the GAT model
    gat_model = GATModel(
        unique_users=train_data["user"].x.size(0),
        item_dim=train_data["item"].x.size(1),
        rating_dim=6,  # Assuming ratings are in the range 1-5
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
        lr=lr,
        use_weighted_loss=use_weighted_loss,
    )

    # Training setup
    logger = CSVLogger(log_dir, name="GAT_training_logs")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=1,
        accelerator="gpu",
    )

    # Train the model
    print("Training GAT model...")
    trainer.fit(gat_model, train_loader, val_loader)
    trainer.test(gat_model, test_loader)

    # Save the model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(gat_model.state_dict(), os.path.join(model_save_path, "gat_model.pt"))
    print(f"Model saved successfully at {model_save_path}")


if __name__ == "__main__":
    train_model()