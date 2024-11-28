import os
from collections import defaultdict

import torch
from encoder.image_encoder.get_image_embeddings import (encode_images,
                                                        encode_texts)
from encoder.image_encoder.prepare_dataset import load_recipe_data_for_graph
from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def prepare_hetero_graph(
    base_dir: str,
    base_text_dir: str,
    log_dir: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    batch_size: int = 32,
    min_interactions: int = 1,
) -> HeteroData:
    """
    Prepare a heterogeneous graph for GNN training, including both "user->item" and "item->user" edges,
    and retaining only users with a minimum number of interactions.

    Args:
        base_dir (str): Directory containing recipe images and comments.
        base_text_dir (str): Directory containing recipe text descriptions and steps.
        log_dir (str): Directory to save training logs.
        processor: AutoImageProcessor: Image processor for encoding images.
        model: ViTForImageClassification: Vision Transformer model for encoding images.
        batch_size (int): Batch size for image encoding.
        min_interactions (int): Minimum number of interactions a user must have to be included.

    Returns:
        HeteroData: Prepared graph with user and item nodes and user-item interaction edges.
    """
    # Load recipe data
    recipes = load_recipe_data_for_graph(base_dir, base_text_dir)

    # Initialize heterogeneous graph
    data = HeteroData()

    ##### Collect user information
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
        and username != "Allrecipes Member"
    }

    # Assign unique IDs to filtered users
    user_ids = {name: idx for idx, name in enumerate(sorted(filtered_usernames))}

    # Prepare user features
    user_features = [
        torch.tensor([user_ids[username]], dtype=torch.long)
        for username in sorted(filtered_usernames)
    ]

    ###### Collect item information
    # Assign unique IDs to items
    item_ids = {name: idx for idx, name in enumerate(sorted(recipe_names))}
    num_items = len(item_ids)

    # Prepare item features
    all_image_paths = []
    image_item_ids = []
    description_texts = []
    description_item_ids = []

    # Collect all image paths and descriptions with their corresponding item IDs
    for recipe in recipes:
        recipe_name = recipe["recipe_name"]
        item_id = item_ids[recipe_name]
        # Collect image paths
        for image_path in recipe["image_paths"]:
            all_image_paths.append(image_path)
            image_item_ids.append(item_id)
        # Collect descriptions
        description_texts.append(recipe["description"])
        description_item_ids.append(item_id)

    ##### Collect ingredients information
    unique_ingredients = set()
    for recipe in recipes:
        unique_ingredients.update(recipe["ingredients"])

    # Map unique ingredients to IDs
    ingredient_ids = {name: idx for idx, name in enumerate(sorted(unique_ingredients))}

    # Encode ingredients into embeddings
    ingredient_texts = sorted(unique_ingredients)
    ingredient_embeddings = encode_texts(ingredient_texts, processor, model)

    ##### Batch process images using encode_images
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_image_embeddings = []
    num_images = len(all_image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Encoding images"):
        batch_image_paths = all_image_paths[i * batch_size : (i + 1) * batch_size]
        batch_embeddings = encode_images(batch_image_paths, processor, model)
        all_image_embeddings.append(batch_embeddings)

    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

    # Aggregate embeddings per item
    item_image_embeddings = defaultdict(list)
    for embedding, item_id in zip(all_image_embeddings, image_item_ids):
        item_image_embeddings[item_id].append(embedding)

    # Compute mean embedding per item
    item_features_list = []
    image_embedding_dim = all_image_embeddings.size(1)
    for item_id in range(num_items):
        embeddings_list = item_image_embeddings[item_id]
        if embeddings_list:
            item_agg_embedding = torch.stack(embeddings_list).mean(dim=0)
        else:
            # If no embeddings are available for this item, use a zero vector
            item_agg_embedding = torch.zeros(image_embedding_dim)
        item_features_list.append(item_agg_embedding)

    item_image_features = torch.stack(item_features_list)

    ###### Batch process descriptions using encode_texts

    all_text_embeddings = []
    num_texts = len(description_texts)
    num_batches = (num_texts + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Encoding texts"):
        batch_texts = description_texts[i * batch_size : (i + 1) * batch_size]
        batch_embeddings = encode_texts(batch_texts, processor, model)
        all_text_embeddings.append(batch_embeddings)

    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    # Map text embeddings to item IDs
    item_text_embeddings = {}
    for embedding, item_id in zip(all_text_embeddings, description_item_ids):
        item_text_embeddings[item_id] = embedding

    # Build the list of text embeddings per item
    text_embedding_dim = all_text_embeddings.size(1)
    item_text_features_list = []
    for item_id in range(num_items):
        if item_id in item_text_embeddings:
            text_embedding = item_text_embeddings[item_id]
        else:
            text_embedding = torch.zeros(text_embedding_dim)
        item_text_features_list.append(text_embedding)

    item_text_features = torch.stack(item_text_features_list)

    # Concatenate image and text embeddings
    item_features = torch.cat([item_image_features, item_text_features], dim=1)

    ###### Collect user-item interactions
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

    ##### Collect item-ingredient interactions
    # Prepare edges between recipes and ingredients
    edge_index_item_to_ingredient = []
    edge_index_ingredient_to_item = []

    for recipe in recipes:
        item_id = item_ids[recipe["recipe_name"]]
        for ingredient in recipe["ingredients"]:
            ingredient_id = ingredient_ids[ingredient]
            # Connect item -> ingredient
            edge_index_item_to_ingredient.append([item_id, ingredient_id])
            # Connect ingredient -> item
            edge_index_ingredient_to_item.append([ingredient_id, item_id])

    # Convert edge lists to tensors
    if edge_index_item_to_ingredient:
        edge_index_item_to_ingredient = (
            torch.tensor(edge_index_item_to_ingredient, dtype=torch.long)
            .t()
            .contiguous()
        )
    else:
        edge_index_item_to_ingredient = torch.empty((2, 0), dtype=torch.long)

    if edge_index_ingredient_to_item:
        edge_index_ingredient_to_item = (
            torch.tensor(edge_index_ingredient_to_item, dtype=torch.long)
            .t()
            .contiguous()
        )
    else:
        edge_index_ingredient_to_item = torch.empty((2, 0), dtype=torch.long)

    ##### Finalize the graph
    # Add user nodes
    data["user"].x = torch.stack(user_features)

    # Add item nodes
    data["item"].x = item_features

    # Add ingredient nodes
    data["ingredient"].x = ingredient_embeddings

    # Add edges and edge attributes
    data["user", "rates", "item"].edge_index = edge_index_user_to_item
    data["user", "rates", "item"].edge_attr = edge_attr_user_to_item

    data["item", "rated_by", "user"].edge_index = edge_index_item_to_user
    data["item", "rated_by", "user"].edge_attr = edge_attr_item_to_user

    # Add edges to the graph
    data["item", "contains", "ingredient"].edge_index = edge_index_item_to_ingredient
    data["ingredient", "included_in", "item"].edge_index = edge_index_ingredient_to_item

    # Save the user and item mappings for later use
    data.user_ids = user_ids
    data.item_ids = item_ids
    data.ingredient_ids = ingredient_ids

    # Save user and item IDs to log file
    log_file_path = os.path.join(log_dir, "user_item_ids.txt")
    with open(log_file_path, "w") as f:
        f.write("User IDs:\n")
        for username, user_id in user_ids.items():
            f.write(f"{username}: {user_id}\n")

        f.write("\nItem IDs:\n")
        for recipe_name, item_id in item_ids.items():
            f.write(f"{recipe_name}: {item_id}\n")

        f.write("\nIngredient IDs:\n")
        for ingredient, ingredient_id in ingredient_ids.items():
            f.write(f"{ingredient}: {ingredient_id}\n")

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
    edge_type_item_ingredient = ("item", "contains", "ingredient")
    edge_type_ingredient_item = ("ingredient", "included_in", "item")

    # User-Item edges
    user_item_edges = data[edge_type_user_item].edge_index.t().contiguous()
    item_user_edges = data[edge_type_item_user].edge_index.t().contiguous()
    assert torch.equal(user_item_edges[:, [1, 0]], item_user_edges)

    # Item-Ingredient edges
    item_ingredient_edges = data[edge_type_item_ingredient].edge_index.t().contiguous()
    ingredient_item_edges = data[edge_type_ingredient_item].edge_index.t().contiguous()
    assert torch.equal(item_ingredient_edges[:, [1, 0]], ingredient_item_edges)

    # Combine user-item edges
    combined_user_item_edges = user_item_edges  # Only one direction is needed
    num_user_item_edges = combined_user_item_edges.size(0)
    user_item_edge_indices = torch.randperm(num_user_item_edges)

    # Combine item-ingredient edges
    combined_item_ingredient_edges = (
        item_ingredient_edges  # Only one direction is needed
    )
    num_item_ingredient_edges = combined_item_ingredient_edges.size(0)
    item_ingredient_edge_indices = torch.randperm(num_item_ingredient_edges)

    # Compute split indices for user-item edges
    user_item_train_end = int(train_ratio * num_user_item_edges)
    user_item_val_end = int((train_ratio + val_ratio) * num_user_item_edges)
    user_item_train_edges = user_item_edge_indices[:user_item_train_end]
    user_item_val_edges = user_item_edge_indices[user_item_train_end:user_item_val_end]
    user_item_test_edges = user_item_edge_indices[user_item_val_end:]

    # Compute split indices for item-ingredient edges
    item_ingredient_train_end = int(train_ratio * num_item_ingredient_edges)
    item_ingredient_val_end = int((train_ratio + val_ratio) * num_item_ingredient_edges)
    item_ingredient_train_edges = item_ingredient_edge_indices[
        :item_ingredient_train_end
    ]
    item_ingredient_val_edges = item_ingredient_edge_indices[
        item_ingredient_train_end:item_ingredient_val_end
    ]
    item_ingredient_test_edges = item_ingredient_edge_indices[item_ingredient_val_end:]

    def create_subgraph(user_item_edges, item_ingredient_edges):
        subgraph = HeteroData()

        # User-Item edges
        selected_user_item_edges = combined_user_item_edges[user_item_edges]
        selected_item_user_edges = selected_user_item_edges[:, [1, 0]]
        subgraph[edge_type_user_item].edge_index = (
            selected_user_item_edges.t().contiguous()
        )
        subgraph[edge_type_user_item].edge_attr = (
            data[edge_type_user_item].edge_attr[user_item_edges].contiguous()
        )
        subgraph[edge_type_item_user].edge_index = (
            selected_item_user_edges.t().contiguous()
        )
        subgraph[edge_type_item_user].edge_attr = (
            data[edge_type_item_user].edge_attr[user_item_edges].contiguous()
        )

        # Item-Ingredient edges
        selected_item_ingredient_edges = combined_item_ingredient_edges[
            item_ingredient_edges
        ]
        selected_ingredient_item_edges = selected_item_ingredient_edges[:, [1, 0]]
        subgraph[edge_type_item_ingredient].edge_index = (
            selected_item_ingredient_edges.t().contiguous()
        )
        subgraph[edge_type_ingredient_item].edge_index = (
            selected_ingredient_item_edges.t().contiguous()
        )

        # Copy node features
        subgraph["user"].x = data["user"].x.contiguous()
        subgraph["item"].x = data["item"].x.contiguous()
        subgraph["ingredient"].x = data["ingredient"].x.contiguous()

        return subgraph

    # Create train, validation, and test graphs
    train_data = create_subgraph(user_item_train_edges, item_ingredient_train_edges)
    val_data = create_subgraph(user_item_val_edges, item_ingredient_val_edges)
    test_data = create_subgraph(user_item_test_edges, item_ingredient_test_edges)

    return train_data, val_data, test_data
