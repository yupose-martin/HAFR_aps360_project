from collections import defaultdict

import torch
from encoder.image_encoder.get_image_embeddings import encode_images
from encoder.image_encoder.prepare_dataset import load_recipe_data
from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification


def prepare_hetero_graph(
    base_dir: str,
    processor: AutoImageProcessor,
    model: ViTForImageClassification,
    batch_size: int = 32,
    min_interactions: int = 5,
) -> HeteroData:
    """
    Prepare a heterogeneous graph for GNN training, including both "user->item" and "item->user" edges,
    and retaining only users with a minimum number of interactions.

    Args:
        base_dir (str): Directory containing recipe data.
        processor: AutoImageProcessor: Image processor for encoding images.
        model: ViTForImageClassification: Vision Transformer model for encoding images.
        batch_size (int): Batch size for image encoding.
        min_interactions (int): Minimum number of interactions a user must have to be included.

    Returns:
        HeteroData: Prepared graph with user and item nodes and user-item interaction edges.
    """
    # Load recipe data
    recipes = load_recipe_data(base_dir)

    # Initialize heterogeneous graph
    data = HeteroData()

    # Collect unique usernames and recipe names
    recipe_names = set()
    usernames = set()
    user_interactions = defaultdict(int)  # To count interactions per user

    for recipe in recipes:
        recipe_names.add(recipe["recipe_name"])
        for comment in recipe["comments"]:
            username = comment["username"]
            usernames.add(username)
            user_interactions[username] += 1

    # Filter users based on min_interactions
    filtered_usernames = {
        username
        for username in usernames
        if user_interactions[username] >= min_interactions
    }

    # Assign unique IDs to filtered users
    user_ids = {name: idx for idx, name in enumerate(sorted(filtered_usernames))}

    # Prepare user features
    user_features = [
        torch.tensor([user_ids[username]], dtype=torch.long)
        for username in sorted(filtered_usernames)
    ]

    # Assign unique IDs to items
    item_ids = {name: idx for idx, name in enumerate(sorted(recipe_names))}
    num_items = len(item_ids)

    # Prepare item features
    all_image_paths = []
    image_item_ids = []

    # Collect all image paths and their corresponding item IDs
    for recipe in recipes:
        recipe_name = recipe["recipe_name"]
        item_id = item_ids[recipe_name]
        for image_path in recipe["image_paths"]:
            all_image_paths.append(image_path)
            image_item_ids.append(item_id)

    # Batch process images using encode_images
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    num_images = len(all_image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Encoding images"):
        batch_image_paths = all_image_paths[i * batch_size : (i + 1) * batch_size]
        batch_embeddings = encode_images(batch_image_paths, processor, model)
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)

    # Aggregate embeddings per item
    item_embeddings = defaultdict(list)
    for embedding, item_id in zip(embeddings, image_item_ids):
        item_embeddings[item_id].append(embedding)

    # Compute mean embedding per item
    item_features_list = []
    embedding_dim = embeddings.size(1)
    for item_id in range(num_items):
        embeddings_list = item_embeddings[item_id]
        if embeddings_list:
            item_embedding = torch.stack(embeddings_list).mean(dim=0)
        else:
            # If no embeddings are available for this item, use a zero vector
            item_embedding = torch.zeros(embedding_dim)
        item_features_list.append(item_embedding)

    item_features = torch.stack(item_features_list)

    # Prepare edge indices and attributes for both directions
    edge_index_user_to_item = []
    edge_attr_user_to_item = []
    edge_index_item_to_user = []
    edge_attr_item_to_user = []

    for recipe in recipes:
        recipe_name = recipe["recipe_name"]
        item_id = item_ids[recipe_name]
        for comment in recipe["comments"]:
            username = comment["username"]
            if username not in user_ids:
                continue  # Skip users who don't meet min_interactions
            rating = comment["rating"]
            user_id = user_ids[username]

            # Add edge for "user -> item"
            edge_index_user_to_item.append([user_id, item_id])
            edge_attr_user_to_item.append([rating])

            # Add edge for "item -> user"
            edge_index_item_to_user.append([item_id, user_id])
            edge_attr_item_to_user.append([rating])

    # Remove items with no interactions if necessary
    interacted_items = set(item_id for _, item_id in edge_index_user_to_item)
    item_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(sorted(interacted_items))
    }
    num_items = len(item_id_mapping)
    item_features = item_features[list(interacted_items)]

    # Update item IDs in edge indices
    edge_index_user_to_item = [
        [user_id, item_id_mapping[item_id]]
        for user_id, item_id in edge_index_user_to_item
    ]
    edge_index_item_to_user = [
        [item_id_mapping[item_id], user_id]
        for item_id, user_id in edge_index_item_to_user
    ]

    # Convert edge lists to tensors
    if edge_index_user_to_item:
        edge_index_user_to_item = (
            torch.tensor(edge_index_user_to_item, dtype=torch.long).t().contiguous()
        )
        edge_attr_user_to_item = torch.tensor(edge_attr_user_to_item, dtype=torch.float)
    else:
        edge_index_user_to_item = torch.empty((2, 0), dtype=torch.long)
        edge_attr_user_to_item = torch.empty((0, 1), dtype=torch.float)

    if edge_index_item_to_user:
        edge_index_item_to_user = (
            torch.tensor(edge_index_item_to_user, dtype=torch.long).t().contiguous()
        )
        edge_attr_item_to_user = torch.tensor(edge_attr_item_to_user, dtype=torch.float)
    else:
        edge_index_item_to_user = torch.empty((2, 0), dtype=torch.long)
        edge_attr_item_to_user = torch.empty((0, 1), dtype=torch.float)

    # Add user nodes
    data["user"].x = torch.stack(user_features)

    # Add item nodes
    data["item"].x = item_features

    # Add edges and edge attributes
    data["user", "rates", "item"].edge_index = edge_index_user_to_item
    data["user", "rates", "item"].edge_attr = edge_attr_user_to_item

    data["item", "rated_by", "user"].edge_index = edge_index_item_to_user
    data["item", "rated_by", "user"].edge_attr = edge_attr_item_to_user

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
    user_item_edges = data[edge_type_user_item].edge_index.t().contiguous()
    item_user_edges = data[edge_type_item_user].edge_index.t().contiguous()

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
        selected_user_item_edges = selected_edges.t().contiguous()
        selected_item_user_edges = selected_edges[:, [1, 0]].t().contiguous()

        # Add user -> item edges
        subgraph[edge_type_user_item].edge_index = selected_user_item_edges
        subgraph[edge_type_user_item].edge_attr = (
            data[edge_type_user_item].edge_attr[edge_indices].contiguous()
        )

        # Add item -> user edges
        subgraph[edge_type_item_user].edge_index = selected_item_user_edges
        subgraph[edge_type_item_user].edge_attr = (
            data[edge_type_item_user].edge_attr[edge_indices].contiguous()
        )

        # Copy node features
        subgraph["user"].x = data["user"].x.contiguous()
        subgraph["item"].x = data["item"].x.contiguous()

        return subgraph

    # Create train, validation, and test graphs
    train_data = create_subgraph(train_edges)
    val_data = create_subgraph(val_edges)
    test_data = create_subgraph(test_edges)

    return train_data, val_data, test_data
