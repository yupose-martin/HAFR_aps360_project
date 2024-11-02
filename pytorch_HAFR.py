# Description: This file contains the implementation of the HAFR model using PyTorch.
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description="Run HAFR.")
    parser.add_argument('--subset_size', type=int, default='999999999',
                        help='Subset used for training,whole dataset contains 676945 items.')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='data',
                        help='Choose a dataset.')
    parser.add_argument('--val_verbose', type=int, default=10,
                        help='Evaluate per X epochs for validation set.')
    parser.add_argument('--test_verbose', type=int, default=10,
                        help='Evaluate per X epochs for test set.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--reg', type=float, default=0.1,
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--reg_image', type=float, default=0.01,
                        help='Regularization for image embeddings.')
    parser.add_argument('--reg_w', type=float, default=1,
                        help='Regularization for mlp w.')
    parser.add_argument('--reg_h', type=float, default=1,
                        help='Regularization for mlp h.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='Use the pretraining weights or not')
    parser.add_argument('--ckpt', type=int, default=10,
                        help='Save the model per X epochs.')
    parser.add_argument('--weight_size', type=int, default=64,
                        help='weight_size')
    parser.add_argument('--save_folder', nargs="?", default='save',
                        help='Choose a save folder in pretrain')
    parser.add_argument('--restore_folder', nargs="?", default='restore',
                        help='Choose a restore folder in pretrain')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, dataset, subset_size=1000,dns = 1):
        self.dataset = dataset
        self.dns = dns
        self.subset_size = subset_size
        self.user_input, self.item_input_pos, self.ingre_input_pos, self.ingre_num_pos, self.image_input_pos, self.labels, self.max_idx = self.sampling(dataset)
        print(f"Maximum index in dataset: {self.max_idx}")

    def sampling(self, dataset):
        user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels = [], [], [], [], [], []
        max_idx = 0
        for idx, (u, i) in enumerate(dataset.trainMatrix.keys()):
            if idx >= self.subset_size:
                break
            user_input.append(u)
            item_input_pos.append(i)
            ingre_input_pos.append(dataset.ingreCodeDict[i])
            ingre_num_pos.append(dataset.ingreNum[i])
            image_input_pos.append(dataset.embImage[i])
            labels.append(1)  # Positive interaction
            max_idx = max(max_idx, idx)
        
        # Generate negative samples
        num_negatives = len(user_input) * self.dns
        for _ in range(num_negatives):
            u = np.random.choice([key[0] for key in dataset.trainMatrix.keys()])
            j = np.random.randint(dataset.num_items)
            while j in dataset.trainList[u] or j in dataset.validTestRatings[u]:
                j = np.random.randint(dataset.num_items)
            user_input.append(u)
            item_input_pos.append(j)
            ingre_input_pos.append(dataset.ingreCodeDict[j])
            ingre_num_pos.append(dataset.ingreNum[j])
            image_input_pos.append(dataset.embImage[j])
            labels.append(0)  # Negative interaction

        return user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels, max_idx

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, idx):
        return (self.user_input[idx], self.item_input_pos[idx], self.ingre_input_pos[idx], self.ingre_num_pos[idx], self.image_input_pos[idx], self.labels[idx])
    
class HAFR(nn.Module):
    def __init__(self, num_users, num_items, num_cold, num_ingredients, image_size, args):
        super(HAFR, self).__init__()
        self.num_items = num_items
        self.num_cold = num_cold
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.dns = args.dns
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_ingredients = num_ingredients
        self.image_size = image_size
        self.reg_image = args.reg_image
        self.reg_w = args.reg_w
        self.reg_h = args.reg_h
        self.weight_size = args.weight_size
        self.trained = args.pretrain

        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(num_items + num_cold, self.embedding_size)
        self.ingredient_embeddings = nn.Embedding(num_ingredients + 1, self.embedding_size)

        # Image weights and bias
        self.W_image = nn.Linear(self.image_size, self.embedding_size)
        self.b_image = nn.Parameter(torch.zeros(1, self.embedding_size))

        # MLP weights and bias
        self.W_concat = nn.Linear(self.embedding_size * 3, self.embedding_size)
        self.b_concat = nn.Parameter(torch.zeros(1, self.embedding_size))

        self.h = nn.Linear(self.embedding_size, 1)

        self.W_att_ingre = nn.Linear(self.embedding_size * 3, self.weight_size)
        self.b_att_ingre = nn.Parameter(torch.zeros(1, self.weight_size))

        self.v = nn.Parameter(torch.ones(self.weight_size, 1))

        self.W_att_com = nn.Linear(self.embedding_size * 2, self.weight_size)
        self.b_att_com = nn.Parameter(torch.zeros(1, self.weight_size))

        self.v_c = nn.Parameter(torch.ones(self.weight_size, 1))

    def forward(self, user_input, item_input, ingre_input, image_input, ingre_num):
        user_embedding = self.user_embeddings(user_input)
        item_embedding = self.item_embeddings(item_input)
        ingre_embedding = self.ingredient_embeddings(ingre_input)
        image_embedding = self.W_image(image_input) + self.b_image

        ingre_att = self._attention_ingredient_level(ingre_embedding, user_embedding, image_embedding, ingre_num)
        item_att = self._attention_id_ingre_image(user_embedding, item_embedding, ingre_att, image_embedding)

        user_item_concat = torch.cat([user_embedding, item_att, user_embedding * item_att], dim=-1)
        hidden_input = self.W_concat(user_item_concat) + self.b_concat
        hidden_output = torch.relu(hidden_input)

        output = self.h(hidden_output)
        # in HAFR paper, they didn't seen to add the sigmoid activation function
        # I think it's okay to add it here or not
        output = torch.sigmoid(output)
        return output

    def _attention_ingredient_level(self, q_, embedding_p, image_embed, item_ingre_num):
        b, n, _ = q_.shape
        tile_p = embedding_p.unsqueeze(1).expand(-1, n, -1)
        tile_image = image_embed.unsqueeze(1).expand(-1, n, -1)
        concat_v = torch.cat([q_, tile_p, tile_image], dim=-1)
        MLP_output = torch.tanh(self.W_att_ingre(concat_v) + self.b_att_ingre)
        A_ = torch.matmul(MLP_output, self.v).squeeze(-1)
        mask = (torch.arange(n, device=q_.device).expand(b, n) < item_ingre_num.unsqueeze(1)).float()
        A = torch.softmax(A_ + (1 - mask) * -1e12, dim=-1).unsqueeze(-1)
        return torch.sum(A * q_, dim=1)

    def _attention_id_ingre_image(self, embedding_p, embedding_q, embedding_ingre_att, image_embed):
        cp1 = torch.cat([embedding_p, embedding_q], dim=-1)
        cp2 = torch.cat([embedding_p, embedding_ingre_att], dim=-1)
        cp3 = torch.cat([embedding_p, image_embed], dim=-1)
        cp = torch.cat([cp1, cp2, cp3], dim=0)
        c_hidden_output = torch.tanh(self.W_att_com(cp) + self.b_att_com)
        c_mlp_output = torch.matmul(c_hidden_output, self.v_c).view(-1, 3)
        B = torch.softmax(c_mlp_output, dim=-1).unsqueeze(-1)
        ce = torch.stack([embedding_q, embedding_ingre_att, image_embed], dim=1)
        return torch.sum(B * ce, dim=1)
    
def train(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    epoch_losses = []  # List to store loss for each epoch
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels = batch
            user_input = user_input.long().to(device)
            item_input_pos = item_input_pos.long().to(device)
            ingre_input_pos = ingre_input_pos.long().to(device)
            image_input_pos = image_input_pos.float().to(device)
            ingre_num_pos = ingre_num_pos.long().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            output = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            training_loss = criterion(output.squeeze(), labels)
            training_loss.backward()
            optimizer.step()
            total_loss += training_loss.item()
        
        avg_training_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_training_loss)  # Store the average loss for this epoch
        print(f"Epoch {epoch+1}, Loss: {avg_training_loss}")
        
        ## Validation to be added here !!!!!!!!!!!!!!!!!!!
    
    return epoch_losses  # Return the list of epoch losses

if __name__ == '__main__':
    args = parse_args()
    print("Arguments: %s" % (args))

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # initialize dataset
    dataset = Dataset(args.path + args.dataset)
    print("Has finished processing dataset")

    # initialize models
    model = HAFR(dataset.num_users, dataset.num_items, dataset.cold_num, dataset.num_ingredients, dataset.image_size, args).to(device)

    # DataLoader
    # totally 676945 items
    # custom_dataset = CustomDataset(dataset, subset_size=9999999999, dns=model.dns)
    custom_dataset = CustomDataset(dataset, subset_size=args.subset_size, dns=model.dns)
    dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # Define your loss function

    # start training
    losses_per_epoch = train(model, dataloader, optimizer, criterion, args.epochs, device)
    # Save the losses_per_epoch to a file
    losses_file_path = os.path.join(args.save_folder, 'losses_per_epoch.npy')
    np.save(losses_file_path, losses_per_epoch)
    print(f"Losses per epoch saved to {losses_file_path}")

    # # To plot the losses later, you can use the following code:
    # import matplotlib.pyplot as plt

    # # Load the losses from the file
    # loaded_losses = np.load(losses_file_path)

    # # Plot the losses
    # plt.plot(loaded_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss per Epoch')
    # plt.show()
    
    # Ensure the save folder exists
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    # Save the model
    save_path = os.path.join(args.save_folder, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # # Load the model
    # load_path = os.path.join(args.restore_folder, 'model.pth')
    # if os.path.exists(load_path):
    #     model.load_state_dict(torch.load(load_path))
    #     print(f"Model loaded from {load_path}")
    # else:
    #     print(f"No model found at {load_path}")
    #     # Ensure the save folder exists
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)