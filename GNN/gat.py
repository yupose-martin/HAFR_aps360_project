import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv


class GATModel(LightningModule):
    def __init__(
        self,
        unique_users: int,
        item_dim: int,
        ingredient_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads_per_layer: int,
        lr: float,
        use_weighted_loss: bool = True,
    ) -> None:
        """
        PyTorch Lightning model for a bipartite graph using GATv2Conv.

        Args:
            unique_users (int): Number of unique user IDs.
            item_dim (int): Dimension of the CLIP embedding for items.
            ingredient_dim (int): Dimension of the CLIP embedding for ingredients.
            hidden_dim (int): Dimension of the hidden representations in GNN layers.
            num_layers (int): Number of GATv2Conv layers.
            num_heads_per_layer (int): Number of attention heads per GATv2Conv layer.
            lr (float): Learning rate for optimization.
            use_weighted_loss (bool): Whether to use weighted loss based on rating frequencies.
        """
        super().__init__()
        self.save_hyperparameters()

        # User embedding: One-hot encoding to dense embedding
        self.user_embed = nn.Embedding(unique_users, hidden_dim)

        # Linear layer to convert item CLIP embeddings to hidden_dim
        self.item_linear = nn.Linear(item_dim, hidden_dim)

        # Linear layer to convert ingredient CLIP embeddings to hidden_dim
        self.ingredient_linear = nn.Linear(ingredient_dim, hidden_dim)

        # GATConv layers for user->item and item->user
        self.gat_user_to_item = nn.ModuleList()
        self.gat_item_to_user = nn.ModuleList()
        self.gat_ingredient_to_item = nn.ModuleList()
        self.gat_item_to_ingredient = nn.ModuleList()
        for i in range(num_layers):
            out_channels = hidden_dim
            self.gat_item_to_user.append(
                GATv2Conv(
                    (-1, -1),
                    out_channels,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    heads=num_heads_per_layer,
                )
            )
            self.gat_user_to_item.append(
                GATv2Conv(
                    (-1, -1),
                    out_channels,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    heads=num_heads_per_layer,
                )
            )
            self.gat_ingredient_to_item.append(
                GATv2Conv(
                    (-1, -1),
                    out_channels,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    heads=num_heads_per_layer,
                )
            )
            self.gat_item_to_ingredient.append(
                GATv2Conv(
                    (-1, -1),
                    out_channels,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    heads=num_heads_per_layer,
                )
            )

        self.user_layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.item_layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.ingredient_layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.lr = lr
        self.use_weighted_loss = use_weighted_loss

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """
        Forward pass for the bipartite GNN.

        Args:
            data (HeteroData): A heterogeneous graph with:
                - 'item': CLIP embeddings as node features.
                - 'user': One-hot encoded features (converted in forward pass).
                - 'edge_index' for 'interaction'.

        Returns:
            dict[str, torch.Tensor]: Updated embeddings for 'user' and 'item' nodes.
        """
        # Initialize user and item node features
        data = data.detach()
        user_indices = data["user"].x.reshape(-1)
        user_embeddings = self.user_embed(user_indices)
        item_embeddings = self.item_linear(data["item"].x)
        ingredient_embeddings = self.ingredient_linear(data["ingredient"].x)

        for (
            gat_item_to_user,
            gat_user_to_item,
            gat_ingredient_to_item,
            gat_item_to_ingredient,
        ) in zip(
            self.gat_item_to_user,
            self.gat_user_to_item,
            self.gat_ingredient_to_item,
            self.gat_item_to_ingredient,
        ):
            # Ingredient -> Item
            initial_item_embeddings = item_embeddings
            item_embeddings = gat_ingredient_to_item(
                (ingredient_embeddings, item_embeddings),
                data["ingredient", "included_in", "item"].edge_index,
            )
            item_embeddings = self.item_layer_norm(F.leaky_relu(item_embeddings))

            # Item -> Ingredient
            ingredient_embeddings = gat_item_to_ingredient(
                (item_embeddings, ingredient_embeddings),
                data["item", "contains", "ingredient"].edge_index,
            )
            ingredient_embeddings = self.ingredient_layer_norm(
                F.leaky_relu(ingredient_embeddings)
            )

            # Item -> User
            user_embeddings = gat_item_to_user(
                (item_embeddings, user_embeddings),
                data["item", "rated_by", "user"].edge_index,
            )
            user_embeddings = self.user_layer_norm(F.leaky_relu(user_embeddings))

            # User -> Item
            item_embeddings = gat_user_to_item(
                (user_embeddings, item_embeddings),
                data["user", "rates", "item"].edge_index,
            )
            item_embeddings = self.item_layer_norm(
                F.leaky_relu(item_embeddings) + initial_item_embeddings
            )

        return {
            "user": user_embeddings,
            "item": item_embeddings,
            "ingredient": ingredient_embeddings,
        }

    def compute_edge_scores(
        self, x_dict: dict, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scores for edges based on source and target node embeddings.

        Args:
            x_dict (dict): Dictionary of node embeddings.
            edge_index (torch.Tensor): Edge indices of shape (2, num_edges).

        Returns:
            torch.Tensor: Predicted scores for edges (e.g., ratings).
        """
        source_embeddings = x_dict["user"][edge_index[0]]
        target_embeddings = x_dict["item"][edge_index[1]]
        scores = (source_embeddings * target_embeddings).sum(dim=-1)  # Dot product
        return scores

    def compute_loss(
        self, predicted_ratings: torch.Tensor, ground_truth_ratings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss for the predicted ratings.

        Args:
            predicted_ratings (torch.Tensor): Predicted ratings.
            ground_truth_ratings (torch.Tensor): Ground truth ratings.

        Returns:
            torch.Tensor: Weighted loss.
        """
        if self.use_weighted_loss:
            # Compute unique values and their counts
            unique_ratings, counts = torch.unique(
                ground_truth_ratings, return_counts=True
            )
            frequencies = counts.float() / counts.sum()

            # Frequency-based weights (inverse frequency)
            inverse_freq_weights = torch.log(1 / frequencies)

            weights_dict = {
                rating.item(): weight
                for rating, weight in zip(unique_ratings, inverse_freq_weights)
            }

            # Map weights to each ground truth rating
            weights = torch.tensor(
                [weights_dict[rating.item()] for rating in ground_truth_ratings],
                device=ground_truth_ratings.device,
            )

        else:
            weights = torch.ones_like(ground_truth_ratings)

        # Compute weighted loss
        squared_errors = (predicted_ratings - ground_truth_ratings) ** 2
        weighted_loss = (weights * squared_errors).mean()
        return weighted_loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """
        Training step for LightningModule.

        Args:
            batch (HeteroData): A batch of heterogeneous graph data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        # Forward pass
        x_dict = self(batch)

        # Compute edge scores
        edge_index = batch["rates"].edge_index
        predicted_ratings = self.compute_edge_scores(x_dict, edge_index)

        # Ground truth ratings
        ground_truth_ratings = (
            batch["user", "rates", "item"].edge_attr.float().reshape(-1)
        )

        # Compute training loss
        weighted_loss = self.compute_loss(predicted_ratings, ground_truth_ratings)

        self.log(
            "train_loss",
            weighted_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch["user"].num_nodes,
        )
        return weighted_loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        """
        Validation step for LightningModule.

        Args:
            batch (HeteroData): A batch of heterogeneous graph data.
            batch_idx (int): Index of the batch.
        """
        # Forward pass
        x_dict = self(batch)

        # Compute edge scores
        edge_index = batch["rates"].edge_index
        predicted_ratings = self.compute_edge_scores(x_dict, edge_index)

        # Ground truth ratings
        ground_truth_ratings = (
            batch["user", "rates", "item"].edge_attr.float().reshape(-1)
        )

        # Compute validation loss
        loss = self.compute_loss(predicted_ratings, ground_truth_ratings)

        # Hard to define batch size for validation
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["user"].num_nodes,
        )
        print(f"Validation loss: {loss}")
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        """
        Test step for LightningModule.

        Args:
            batch (HeteroData): A batch of heterogeneous graph data.
            batch_idx (int): Index of the batch.
        """
        # Forward pass
        x_dict = self(batch)

        # Compute edge scores
        edge_index = batch["rates"].edge_index
        predicted_ratings = self.compute_edge_scores(x_dict, edge_index)

        # Ground truth ratings
        ground_truth_ratings = (
            batch["user", "rates", "item"].edge_attr.float().reshape(-1)
        )

        for i in range(len(predicted_ratings)):
            print(
                f"Predicted rating: {predicted_ratings[i]}, Ground truth rating: {ground_truth_ratings[i]}"
            )

        # Compute test loss
        loss = self.compute_loss(predicted_ratings, ground_truth_ratings)

        # Hard to define batch size for test
        # self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        print(f"Test loss: {loss}")
        return

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for the model.

        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
