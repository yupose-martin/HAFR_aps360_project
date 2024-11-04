# Perform Latent Dirichlet Allocation on the User-Recipe interaction matrix used for recommendation.
# Note that only the User-Recipe interaction matrix is used for LDA, and the recipe ingredients and images are ignored.
from matplotlib.pylab import f
import numpy as np
import os
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Dataset import Dataset


class LDAModel:
    def __init__(self, num_topics=10, random_state=42):
        """
        Initialize the LDA model.

        Parameters:
        - num_topics: Number of latent topics to model.
        - random_state: Random state for reproducibility.
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics, random_state=self.random_state
        )
        self.user_topic_dist = None
        self.topic_recipe_dist = None

    def fit(self, interaction_matrix):
        """
        Train the LDA model using the user-recipe interaction matrix.

        Parameters:
        - interaction_matrix: Numpy array of shape (num_users, num_items)
        """
        # Fit the LDA model
        self.lda_model.fit(interaction_matrix)

        # Compute User-Topic Distribution
        self.user_topic_dist = self.lda_model.transform(interaction_matrix)
        # Shape: (num_users, num_topics)

        # Compute Topic-Recipe Distribution
        self.topic_recipe_dist = self.lda_model.components_
        # Shape: (num_topics, num_items)
        # Normalize to get probabilities
        self.topic_recipe_dist /= self.topic_recipe_dist.sum(axis=1)[:, np.newaxis]

    def predict_proba(self, user_item_pairs):
        """
        Predict the probability that users like the given recipes.

        Parameters:
        - user_item_pairs: List of tuples (user_id, item_id)

        Returns:
        - probabilities: Numpy array of predicted probabilities
        """
        probabilities = []
        for user_id, item_id in user_item_pairs:
            # get user-topic distribution
            user_topic_dist = self.user_topic_dist[user_id]

            # get topic-recipe distribution
            if item_id >= self.topic_recipe_dist.shape[1]:
                # item_id is out of bounds
                probabilities.append(0)
                continue
            topic_recipe_dist = self.topic_recipe_dist[:, item_id]
            topic_recipe_dist = topic_recipe_dist / topic_recipe_dist.sum()

            # compute cosine similarity
            similarity = np.dot(user_topic_dist, topic_recipe_dist) / (
                np.linalg.norm(user_topic_dist) * np.linalg.norm(topic_recipe_dist)
            )

            # use similarity as probability
            probabilities.append(similarity)

        return np.array(probabilities)

    def predict(self, user_item_pairs, threshold):
        """
        Predict whether users like the given recipes based on a threshold.

        Parameters:
        - user_item_pairs: List of tuples (user_id, item_id)
        - threshold: Probability threshold for classification

        Returns:
        - predictions: Numpy array of predicted labels (0 or 1)
        """
        probabilities = self.predict_proba(user_item_pairs)
        predictions = (probabilities >= threshold).astype(int)
        return predictions

    def evaluate(self, user_item_pairs, true_labels, threshold=0.6):
        """
        Evaluate the model's performance.

        Parameters:
        - user_item_pairs: List of tuples (user_id, item_id)
        - true_labels: Numpy array of true labels (0 or 1)
        - threshold: Probability threshold for classification

        Returns:
        - metrics: Dictionary containing accuracy, precision, recall, and F1 score
        """
        predictions = self.predict(user_item_pairs, threshold)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Dirichlet Allocation")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--neg_samples_ratio",
        type=int,
        default=1,
        help="Ratio of negative samples to positive samples",
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        default=3,
        help="Number of latent topics to model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num_topics = args.num_topics

    ## Load the user-recipe interaction matrix
    dataset = Dataset(os.path.join(args.data_path, args.dataset))

    ## Initialize the LDA model
    lda_model = LDAModel(num_topics=num_topics)
    interaction_matrix = dataset.trainMatrix

    ## Fit the LDA model
    lda_model.fit(interaction_matrix)

    print("LDA Model Training Complete")

    ## Evaluate the LDA model
    user_item_pairs = []
    true_labels = []

    # Prepare positive user-item pairs
    for user_id, items in enumerate(dataset.testRatings):
        for item_id in items:
            if item_id >= dataset.num_items:
                continue
            user_item_pairs.append((user_id, item_id))
            true_labels.append(1)

    num_positives = len(user_item_pairs)

    print(f"Number of positive user-item pairs to test: {num_positives}")

    # Sample negative items
    for user_id, items in enumerate(dataset.testNegatives):
        effective_items = [item for item in items if item < dataset.num_items]
        item_ids = np.random.choice(
            effective_items,
            np.min(
                [len(dataset.testRatings[user_id]) * args.neg_samples_ratio, len(items)]
            ),
            replace=False,
        )

        for item_id in item_ids:
            user_item_pairs.append((user_id, item_id))
            true_labels.append(0)

    user_item_pairs = np.array(user_item_pairs)

    num_negatives = len(user_item_pairs) - num_positives
    print(f"Number of negative user-item pairs to test: {num_negatives}")

    true_labels = np.array(true_labels)

    metrics = lda_model.evaluate(user_item_pairs, true_labels)

    print("Evaluation Metrics:")
    print(metrics)
