import torch
from encoder.image_encoder.get_image_embeddings import encode_image
from encoder.image_encoder.prepare_dataset import load_recipe_data
from torch_geometric.data import HeteroData
from transformers import AutoImageProcessor, ViTForImageClassification


def prepare_hetero_graph(
    base_dir: str, processor: AutoImageProcessor, model: ViTForImageClassification
) -> HeteroData:
    """
    Prepare a heterogeneous graph for GNN training.

    Args:
        base_dir (str): Directory containing recipe data.
        processor: Image processor for encoding images.
        model: Vision Transformer model for encoding images.

    Returns:
        HeteroData: Prepared graph with user and item nodes and user-item interaction edges.
    """
    # Load recipe data
    recipes = load_recipe_data(base_dir)

    # Initialize heterogeneous graph
    data = HeteroData()

    # User and item node preparation
    user_ids = {}  # Map usernames to unique IDs
    item_ids = {}  # Map recipe names to unique IDs
    user_features = []
    item_features = []
    edge_index_user_to_item = []  # For "user -> item"
    edge_index_item_to_user = []  # For "item -> user"
    edge_attr_user_to_item = []  # For attributes of "user -> item"
    edge_attr_item_to_user = []  # For attributes of "item -> user"

    for recipe in recipes:
        print(f"Processing recipe: {recipe['recipe_name']}")
        recipe_name = recipe["recipe_name"]

        # Assign a unique ID to the item (recipe) if not already assigned
        if recipe_name not in item_ids:
            item_id = len(item_ids)
            item_ids[recipe_name] = item_id

            # Encode images for the recipe and calculate their mean embedding
            embeddings = []
            for image_path in recipe["image_paths"]:
                embedding = encode_image(image_path, processor, model).squeeze(0)
                embeddings.append(embedding)

            # Use the mean of the embeddings if there are multiple images
            if embeddings:
                mean_embedding = torch.stack(embeddings).mean(dim=0)
                item_features.append(mean_embedding)

        for comment in recipe["comments"]:
            username = comment["username"]
            rating = comment["rating"]

            # Assign a unique ID to the user if not already assigned
            if username not in user_ids:
                user_id = len(user_ids)
                user_ids[username] = user_id
                user_features.append(torch.tensor([user_id], dtype=torch.long))

            # Create edge between user and item
            user_id = user_ids[username]
            item_id = item_ids[recipe_name]

            # Add edge for "user -> item"
            edge_index_user_to_item.append([user_id, item_id])
            edge_attr_user_to_item.append([rating])

            # Add edge for "item -> user"
            edge_index_item_to_user.append([item_id, user_id])
            edge_attr_item_to_user.append([rating])

    # Add user nodes
    data["user"].x = torch.stack(user_features)

    # Add item nodes
    data["item"].x = torch.stack(item_features)

    # Add edges and edge attributes
    if edge_index_user_to_item:
        data["user", "rates", "item"].edge_index = (
            torch.tensor(edge_index_user_to_item, dtype=torch.long).t().contiguous()
        )
        data["user", "rates", "item"].edge_attr = torch.tensor(
            edge_attr_user_to_item, dtype=torch.float
        )

    if edge_index_item_to_user:
        data["item", "rated_by", "user"].edge_index = (
            torch.tensor(edge_index_item_to_user, dtype=torch.long).t().contiguous()
        )
        data["item", "rated_by", "user"].edge_attr = torch.tensor(
            edge_attr_item_to_user, dtype=torch.float
        )

    return data


def split_hetero_graph(data: HeteroData, train_ratio=0.8, val_ratio=0.1):
    """
    Split a HeteroData graph into train, validation, and test sets,
    ensuring that bidirectional edges are placed in the same set.

    Args:
        data (HeteroData): The input heterogeneous graph.
        train_ratio (float): Proportion of edges to use for training.
        val_ratio (float): Proportion of edges to use for validation.

    Returns:
        (HeteroData, HeteroData, HeteroData): Train, validation, and test graphs.
    """
    edge_type_user_item = ("user", "rates", "item")
    edge_type_item_user = ("item", "rated_by", "user")

    # Get the bidirectional edges for user -> item and item -> user
    user_item_edges = data[edge_type_user_item].edge_index.t()
    item_user_edges = data[edge_type_item_user].edge_index.t()

    # Assert that the edges are exactly mirrored
    assert torch.equal(user_item_edges[:, [1, 0]], item_user_edges)

    # Combine user-item and item-user edges into unique pairs
    combined_edges = user_item_edges  # Only one direction is needed to represent pairs

    # Shuffle and split indices
    num_edges = combined_edges.size(0)
    edge_indices = torch.randperm(num_edges)
    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)

    train_edges = edge_indices[:train_end]
    val_edges = edge_indices[train_end:val_end]
    test_edges = edge_indices[val_end:]

    # Helper function to create subgraphs
    def create_subgraph(edge_indices):
        subgraph = HeteroData()

        # Subset edges
        selected_edges = combined_edges[edge_indices]
        selected_user_item_edges = selected_edges.t()
        selected_item_user_edges = selected_edges[:, [1, 0]].t()

        # Add user -> item edges
        subgraph[edge_type_user_item].edge_index = selected_user_item_edges
        subgraph[edge_type_user_item].edge_attr = data[edge_type_user_item].edge_attr[
            edge_indices
        ]

        # Add item -> user edges
        subgraph[edge_type_item_user].edge_index = selected_item_user_edges
        subgraph[edge_type_item_user].edge_attr = data[edge_type_item_user].edge_attr[
            edge_indices
        ]

        # Copy node features
        subgraph["user"].x = data["user"].x
        subgraph["item"].x = data["item"].x

        return subgraph

    # Create train, validation, and test graphs
    train_data = create_subgraph(train_edges)
    val_data = create_subgraph(val_edges)
    test_data = create_subgraph(test_edges)

    return train_data, val_data, test_data
