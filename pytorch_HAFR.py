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
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='data',
                        help='Choose a dataset.')
    parser.add_argument('--val_verbose', type=int, default=10,
                        help='Evaluate per X epochs for validation set.')
    parser.add_argument('--test_verbose', type=int, default=10,
                        help='Evaluate per X epochs for test set.')
    parser.add_argument('--batch_size', type=int, default=64,  # Reduced batch size
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=2,  # Reduced number of epochs
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
    parser.add_argument('--lr', type=float, default=0.05,
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
    def __init__(self, dataset, subset_size=1000):  # Add subset_size parameter
        self.dataset = dataset
        self.subset_size = subset_size
        self.user_input, self.item_input_pos, self.ingre_input_pos, self.ingre_num_pos, self.image_input_pos = self.sampling(dataset)

    def sampling(self, dataset):
        user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos = [], [], [], [], []
        for idx, (u, i) in enumerate(dataset.trainMatrix.keys()):
            if idx >= self.subset_size:  # Limit the subset size
                break
            user_input.append(u)
            item_input_pos.append(i)
            ingre_input_pos.append(dataset.ingreCodeDict[i])
            ingre_num_pos.append(dataset.ingreNum[i])
            image_input_pos.append(dataset.embImage[i])
        return user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, idx):
        return (self.user_input[idx], self.item_input_pos[idx], self.ingre_input_pos[idx], self.ingre_num_pos[idx], self.image_input_pos[idx])

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

        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(num_items + num_cold, self.embedding_size)
        self.ingredient_embeddings = nn.Embedding(num_ingredients + 1, self.embedding_size)

        # Image weights and bias
        self.W_image = nn.Linear(image_size, self.embedding_size)
        self.b_image = nn.Parameter(torch.zeros(1, self.embedding_size))

        # MLP weights and bias
        self.W_concat = nn.Linear(self.embedding_size * 3, self.embedding_size)
        self.b_concat = nn.Parameter(torch.zeros(1, self.embedding_size))

        self.h = nn.Linear(self.embedding_size, 1)

        # Attention weights and bias
        self.W_att_ingre = nn.Linear(self.embedding_size * 3, self.weight_size)
        self.b_att_ingre = nn.Parameter(torch.zeros(1, self.weight_size))
        self.v = nn.Parameter(torch.ones(self.weight_size, 1))

        self.W_att_com = nn.Linear(self.embedding_size * 2, self.weight_size)
        self.b_att_com = nn.Parameter(torch.zeros(1, self.weight_size))
        self.v_c = nn.Parameter(torch.ones(self.weight_size, 1))

    def forward(self, user_input, item_input, ingre_input, image_input, ingre_num):
        user_input = user_input.long()
        item_input = item_input.long()
        ingre_input = ingre_input.long()
        image_input = image_input.float()
        ingre_num = ingre_num.long()

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
        return output

    def _attention_ingredient_level(self, q_, embedding_p, image_embed, item_ingre_num):
        b, n, _ = q_.shape
        tile_p = embedding_p.unsqueeze(1).expand(b, n, -1)
        tile_image = image_embed.unsqueeze(1).expand(b, n, -1)
        concat_v = torch.cat([q_, tile_p, tile_image], dim=2)
        MLP_output = torch.tanh(self.W_att_ingre(concat_v.view(b * n, -1)) + self.b_att_ingre)
        A_ = MLP_output @ self.v
        A_ = A_.view(b, n)
        mask = (torch.arange(n).expand(b, n) < item_ingre_num.unsqueeze(1)).float()
        A_ = A_ * mask + (1 - mask) * -1e12
        A = torch.softmax(A_, dim=1).unsqueeze(2)
        return (A * q_).sum(1)

    def _attention_id_ingre_image(self, embedding_p, embedding_q, embedding_ingre_att, image_embed):
        b = embedding_p.shape[0]
        cp1 = torch.cat([embedding_p, embedding_q], dim=1)
        cp2 = torch.cat([embedding_p, embedding_ingre_att], dim=1)
        cp3 = torch.cat([embedding_p, image_embed], dim=1)
        cp = torch.cat([cp1, cp2, cp3], dim=0)
        c_hidden_output = torch.tanh(self.W_att_com(cp) + self.b_att_com)
        c_mlp_output = c_hidden_output @ self.v_c
        B = torch.softmax(c_mlp_output.view(b, -1), dim=1).unsqueeze(2)
        ce = torch.stack([embedding_q, embedding_ingre_att, image_embed], dim=1)
        return (B * ce).sum(1)

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos = batch
            user_input = user_input.long()
            item_input_pos = item_input_pos.long()
            ingre_input_pos = ingre_input_pos.long()
            image_input_pos = image_input_pos.float()
            ingre_num_pos = ingre_num_pos.long()

            optimizer.zero_grad()
            output = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            print(output)  # Access the output to avoid compile error
            loss = criterion(output, torch.ones_like(output))  # Define your target
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos = batch
            user_input = user_input.long()
            item_input_pos = item_input_pos.long()
            ingre_input_pos = ingre_input_pos.long()
            image_input_pos = image_input_pos.float()
            ingre_num_pos = ingre_num_pos.long()

            output = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            print(output)
            # Compute metrics

if __name__ == '__main__':
    args = parse_args()
    print("Arguments: %s" % (args))

    # initialize dataset
    dataset = Dataset(args.path + args.dataset)
    print("Has finished processing dataset")

    # initialize models
    model = HAFR(dataset.num_users, dataset.num_items, dataset.cold_num, dataset.num_ingredients, dataset.image_size, args)

    # DataLoader
    custom_dataset = CustomDataset(dataset, subset_size=1000)  # Use a smaller subset of data
    dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # Define your loss function

    # start training
    train(model, dataloader, optimizer, criterion, args.epochs)

    # start evaluation
    evaluate(model, dataloader)