import scipy.sparse as sp
import numpy as np
from time import time
import pickle
import datetime

class Dataset(object):
	'''
	Loading the data file
		trainMatrix: load rating records as sparse matrix for class Data
		trianList: load rating records as list to speed up user's feature retrieval
		testRatings: load leave-one-out rating test for class Evaluate
		testNegatives: sample the items not rated by user
	'''

	def __init__(self, path = "../Data/data"):
		'''
		Constructor
		'''
		self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
		self.trainList = self.load_training_file_as_list(path + ".train.rating")
		self.testRatings, self.test_users = self.load_valid_file_as_list(path + ".test.rating")
		self.testNegatives = self.load_negative_file(path + ".test.negative")
		assert len(self.testRatings) == len(self.testNegatives)
		self.num_users, self.num_items = self.trainMatrix.shape	
		self.ingreCodeDict = np.load(path+"_ingre_code_file.npy")
		self.embImage = np.load(path + "_image_features_float.npy")
		self.image_size = self.embImage.shape[1]
	
		self.validRatings, self.valid_users = self.load_valid_file_as_list(path + ".valid.rating")
		self.validNegatives = self.load_negative_file(path + ".valid.negative")
		self.validTestRatings = self.load_valid_test_file_as_dict(path+".valid.rating", path+".test.rating")
		self.num_ingredients = 33147
		self.cold_list, self.cold_num, self.train_item_list = self.get_cold_start_item_num()
		self.ingreNum = self.load_id_ingre_num(path+"_id_ingre_num_file")
	


	def load_valid_test_file_as_dict(self, valid_file, test_file):
		validTestRatings = {}
		for u in range(self.num_users):
				validTestRatings[u] = set()
		fv = open(valid_file, "r")
		for line in fv:
				arr = line.split("\t")
				u, i = int(arr[0]), int(arr[1])
				validTestRatings[u].add(i)
		fv.close()
		ft = open(test_file, "r")
		for line in ft:
				arr = line.split("\t")
				u, i = int(arr[0]), int(arr[1])
				validTestRatings[u].add(i)
		ft.close()
		return validTestRatings


	def get_cold_start_item_num(self):
		train_item_list = []
		for i_list in self.trainList:
				train_item_list.extend(i_list)
		test_item_list = []
		for r in self.testRatings:
				test_item_list.extend(r)
		valid_item_list = []
		for r in self.validRatings:
				valid_item_list.extend(r)
		c_list = list((set(test_item_list) | set(valid_item_list))- set(train_item_list))
		t_list = list(set(train_item_list))
		return c_list, len(c_list), len(t_list)
	

	def load_image(self, filename):
		fr = open(filename, 'rb')
		image_feature_dict_from_pickle = pickle.load(fr)
		fr.flush()
		fr.close()
		return image_feature_dict_from_pickle

	def load_id_ingre_code(self, filename):
		fr = open(filename, 'rb')
		dict_from_pickle = pickle.load(fr)
		fr.flush()
		fr.close()
		return dict_from_pickle

	def load_id_ingre_num(self, filename):
		fr = open(filename, "r")
		ingreNumDict = {}
		for line in fr:
			arr = line.strip().split("\t")
			ingreNumDict[int(arr[0])] = int(arr[1])
		return ingreNumDict


	def load_rating_file_as_list(self, filename):
		ratingList = []
		with open(filename, "r") as f:
			line = f.readline()
			while line != None and line != "":
				arr = line.split("\t")
				user, item = int(arr[0]), int(arr[1])
				ratingList.append([user, item])
				line = f.readline()
		return ratingList

	def load_negative_file(self, filename):
		negativeList = []
		with open(filename, "r") as f:
			line = f.readline()
			while line != None and line != "":
				arr = line.split("\t")
				negatives = []
				for x in arr[1: ]:
					negatives.append(int(x))
				negativeList.append(negatives)
				line = f.readline()
		return negativeList

	def load_training_file_as_matrix(self, filename):
		num_users, num_items = 0, 0
		with open(filename, "r") as f:
			line = f.readline()
			while line != None and line != "":
				arr = line.split("\t")
				u, i = int(arr[0]), int(arr[1])
				num_users = max(num_users, u)
				num_items = max(num_items, i)
				line = f.readline()
		
		mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
		with open(filename, "r") as f:
			line = f.readline()
			while line != None and line != "":
				arr = line.split("\t")
				user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
				if (rating > 0):
					mat[user, item] = 1.0
				line = f.readline()
		return mat

	def load_training_file_as_list(self, filename):
		# Get number of users and items
		u_ = 0
		lists, items = [], []
		with open(filename, "r") as f:
			line = f.readline()
			index = 0
			while line != None and line != "":
				arr = line.split("\t")
				u, i = int(arr[0]), int(arr[1])
				if u_ < u:
					index = 0
					lists.append(items)
					items = []
					u_ += 1
				index += 1
				items.append(i)
				line = f.readline()
		lists.append(items)
		return lists

	def load_valid_file_as_list(self, filename):
		# Get number of users and items
		lists, items, user_list = [], [], []
		with open(filename, "r") as f:
			line = f.readline()
			index = 0
			u_ = int(line.split("\t")[0])
			while line != None and line != "":
				arr = line.split("\t")
				u, i = int(arr[0]), int(arr[1])
				if u_ < u:
					index = 0
					lists.append(items)
					user_list.append(u_)
					items = []
					u_ = u
				index += 1
				items.append(i)
				line = f.readline()
		lists.append(items)
		user_list.append(u)
		return lists, user_list


# Necessary imports
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
'''Dataset,''' 
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set hyperparameters
def get_random_hyperparameters(Try_time):
    params = {}
    params['try'] = Try_time
    params['subset_size'] = 300
    params['path'] = 'Data/'
    params['dataset'] = 'data'
    params['val_verbose'] = 1  # Evaluate every epoch
    params['test_verbose'] = 1  # Evaluate every epoch
    params['batch_size'] = 256#512 #random.choice([256, 512])
    params['epochs'] = 5 #random.randint(50, 50)
    params['embed_size'] = 16#64 #random.choice([16, 32, 64, 128, 256])
    params['dns'] = 1  # Number of negative samples
    params['reg'] = 0.07674#0.05#0.1#random.uniform(0.01, 0.1)
    params['reg_image'] = 0.03204#0.01 #random.uniform(0.01, 0.1)
    params['reg_w'] = 0.22558#1 #random.uniform(0.1, 1)
    params['reg_h'] = 0.19225#1 #random.uniform(0.1, 1)
    params['lr'] = 0.00741#0.01 #random.uniform(0.00001, 0.01)
    params['pretrain'] = 1  # Use pretraining weights or not
    params['ckpt'] = 10  # Save the model every ckpt epochs
    params['weight_size'] = 256#64 #random.choice([16, 32, 64, 128, 256])
    params['save_folder'] = 'save'
    params['restore_folder'] = 'restore'
    params['check_frequency'] = 10
    return params

class CustomDataset(Dataset):

    def __init__(self, dataset, subset_size=1000, dns=1, data_type='train'):
        self.dataset = dataset
        self.dns = dns
        self.subset_size = subset_size
        self.data_type = data_type  # 'train', 'validation', 'test'
        self.user_input, self.item_input_pos, self.ingre_input_pos, self.ingre_num_pos, \
        self.image_input_pos, self.labels, self.max_idx = self.sampling()
        print(f"Maximum index in {self.data_type} dataset: {self.max_idx}")
    
    def sampling(self):
        user_input = []
        item_input_pos = []
        ingre_input_pos = []
        ingre_num_pos = []
        image_input_pos = []
        labels = []
        max_idx = 0
        idx = 0

        if self.data_type == 'train':
            data_source = self.dataset.trainMatrix.keys()
        elif self.data_type == 'validation':
            data_source = []
            for user, items in zip(self.dataset.valid_users, self.dataset.validRatings):
                for item in items:
                    data_source.append((user, item))
        elif self.data_type == 'test':
            data_source = []
            for user, items in zip(self.dataset.test_users, self.dataset.testRatings):
                for item in items:
                    data_source.append((user, item))
        else:
            raise ValueError("data_type must be 'train', 'validation', or 'test'")


        # Positive samples
        #for idx, (u, i) in enumerate(data_source):
        print(f"Creating positive samples for {self.data_type} dataset...")
        for idx, (u, i) in enumerate(tqdm(data_source, desc=f"Processing {self.data_type} data")):
            if idx >= self.subset_size:
                break
            user_input.append(u)
            item_input_pos.append(i)
            ingre_input_pos.append(self.dataset.ingreCodeDict[i])
            ingre_num_pos.append(self.dataset.ingreNum[i])
            image_input_pos.append(self.dataset.embImage[i])
            labels.append(1)  # Positive interaction
            max_idx = max(max_idx, idx)

        # Negative sampling only for training
        if self.data_type == 'train':
            # Negative sampling
            print(f"Generating negative samples for {self.data_type} dataset...")
            users = set(user_input)  # Unique users from the positive samples
            all_items = set(range(self.dataset.num_items))

            for u in tqdm(users, desc="Negative sampling per user"):
                # Get the items the user has liked (positive interactions)
                L_u = set(self.dataset.trainList[u])
                
                # Items the user has not liked
                negative_items_u = list(all_items - L_u)
                
                # Number of items the user has liked
                k = len(L_u)
                
                # If the user hasn't liked any items, skip
                if k == 0 or not negative_items_u:
                    continue
                
                # If k is greater than available negative items, adjust k
                k = min(k, len(negative_items_u))
                
                # Randomly select k items the user hasn't liked
                sampled_negatives = np.random.choice(negative_items_u, size=k, replace=False)
                
                # Extend the input lists with negative samples
                user_input.extend([u] * k)
                item_input_pos.extend(sampled_negatives)
                ingre_input_pos.extend([self.dataset.ingreCodeDict[j] for j in sampled_negatives])
                ingre_num_pos.extend([self.dataset.ingreNum[j] for j in sampled_negatives])
                image_input_pos.extend([self.dataset.embImage[j] for j in sampled_negatives])
                labels.extend([0] * k)  # Negative interaction

        
        return user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels, max_idx

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, idx):
        return (
            self.user_input[idx],
            self.item_input_pos[idx],
            self.ingre_input_pos[idx],
            self.ingre_num_pos[idx],
            self.image_input_pos[idx],
            self.labels[idx]
        )


# Adjust the HAFR model class
class HAFR(nn.Module):
    def __init__(self, num_users, num_items, num_cold, num_ingredients, image_size, params):
        super(HAFR, self).__init__()
        self.num_items = num_items
        self.num_cold = num_cold
        self.num_users = num_users
        self.embedding_size = params['embed_size']
        self.learning_rate = params['lr']
        self.reg = params['reg']
        self.dns = params['dns']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.num_ingredients = num_ingredients
        self.image_size = image_size
        self.reg_image = params['reg_image']
        self.reg_w = params['reg_w']
        self.reg_h = params['reg_h']
        self.weight_size = params['weight_size']
        self.trained = params['pretrain']

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
        return output, user_embedding, item_embedding, ingre_embedding

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

def hafr_loss(outputs, labels, user_embedding, item_embedding, ingre_embedding, model):
    # Separate positive and negative samples based on labels
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    print(f"outputs: {outputs.shape}")

    output_pos = outputs[pos_mask]
    output_neg = outputs[neg_mask]
    
    print(f"output_pos: {output_pos.shape}")
    print(f"output_neg: {output_neg.shape}")

    # # Ensure the number of positive and negative samples are equal by repeating the smaller tensor
    # if output_pos.size(0) > 0 and output_neg.size(0) > 0:
    #     if output_pos.size(0) > output_neg.size(0):
    #         output_neg = output_neg.unsqueeze(0).repeat((output_pos.size(0) // output_neg.size(0)) + 1)[:output_pos.size(0)]
    #     elif output_neg.size(0) > output_pos.size(0):
    #         output_pos = output_pos.unsqueeze(1).repeat((output_neg.size(0) // output_pos.size(0)) + 1, 1)[:output_neg.size(0)]
    # else:
    #     return torch.tensor(0.0, requires_grad=True)

    # Calculate the softplus loss
    # result = output_pos - output_neg
    # loss = torch.sum(torch.nn.functional.softplus(-result))

    # Regularization terms
    reg_loss = model.reg * (
        torch.sum(user_embedding ** 2) + torch.sum(item_embedding[pos_mask] ** 2) + torch.sum(ingre_embedding[pos_mask] ** 2) +
        torch.sum(item_embedding[neg_mask] ** 2) + torch.sum(ingre_embedding[neg_mask] ** 2)
    ) + model.reg_image * torch.sum(model.W_image.weight ** 2) + model.reg_w * torch.sum(model.W_concat.weight ** 2) + model.reg_h * torch.sum(model.h.weight ** 2)

    return loss + reg_loss

# Training function with accuracy
def train(model, train_loader, validation_loader, optimizer, criterion, epochs, device):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Initialize best validation loss to infinity
    best_val_accuracy = 0
    best_model = None  # To store the best model
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        #for batch in train_loader:
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        for batch in train_loader_tqdm:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels = batch
            user_input = user_input.long().to(device)
            item_input_pos = item_input_pos.long().to(device)
            ingre_input_pos = ingre_input_pos.long().to(device)
            image_input_pos = image_input_pos.float().to(device)
            ingre_num_pos = ingre_num_pos.long().to(device)
            output, _, _, _ = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)

            optimizer.zero_grad()
            output, user_embedding, item_embedding, ingre_embedding = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            loss = hafr_loss(output, labels, user_embedding, item_embedding, ingre_embedding, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(output.squeeze())
            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # Update the progress bar with current loss
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_training_loss = total_loss / len(train_loader)
        training_accuracy = correct_predictions / total_samples
        train_losses.append(avg_training_loss)
        train_accuracies.append(training_accuracy)
        #print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}")

        # Validation
        if (epoch + 1) % params['val_verbose'] == 0:
            val_loss, val_accuracy = validate(model, validation_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            #print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Check if this is the best validation loss so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.4f} and accuracy {val_accuracy:.4f}")
                
                # Save the model
                save_path = os.path.join(params['save_folder'], f"best_model_{params['try']}.pth")
                torch.save(best_model.state_dict(), save_path)
                #print(f"Model saved to {save_path}")
                
        if (epoch + 1) % params['check_frequency'] == 0:
            # Save the model
            save_path = os.path.join(params['save_folder'], f"model_{params['try']}_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            #print(f"Model saved to {save_path}")

            # Save the losses
            losses_file_path = os.path.join(params['save_folder'], f"losses_per_epoch_{params['try']}.npy")
            np.save(losses_file_path, {'train_losses': train_losses, 'val_losses': val_losses, "train_accuracies": train_accuracies, "val_accuracies": val_accuracies}, allow_pickle=True)
            #print(f"Losses per epoch saved to {losses_file_path}")

            #plot the training curve
            save_training_curve(train_losses, train_accuracies, val_losses, val_accuracies)
                
    # After training, ensure that a best_model was found
    if best_model is None:
        print("No improvement in validation loss was observed during training.")
        best_model = model  # Fallback to the last model
    else:
        print("Best model was saved successfully.")

    return train_losses, train_accuracies, val_losses, val_accuracies, best_model

# Validation function with accuracy
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels = batch
            user_input = user_input.long().to(device)
            item_input_pos = item_input_pos.long().to(device)
            ingre_input_pos = ingre_input_pos.long().to(device)
            image_input_pos = image_input_pos.float().to(device)
            ingre_num_pos = ingre_num_pos.long().to(device)
            labels = labels.float().to(device)

            output, user_embedding, item_embedding, ingre_embedding = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            loss = criterion(output.squeeze(), labels)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(output.squeeze())
            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# Test function with accuracy
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, labels = batch
            user_input = user_input.long().to(device)
            item_input_pos = item_input_pos.long().to(device)
            ingre_input_pos = ingre_input_pos.long().to(device)
            image_input_pos = image_input_pos.float().to(device)
            ingre_num_pos = ingre_num_pos.long().to(device)
            labels = labels.float().to(device)

            output, user_embedding, item_embedding, ingre_embedding = model(user_input, item_input_pos, ingre_input_pos, image_input_pos, ingre_num_pos)
            loss = criterion(output.squeeze(), labels)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(output.squeeze())
            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def save_training_curve(train_losses, train_accuracies, val_losses, val_accuracies):
    # Plot the training and validation losses and accuracies
    plt.figure(figsize=(10, 4))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot to a file
    plot_file_path = os.path.join(params['save_folder'], f"loss_accuracy_plot_{params['try']}.png")
    plt.savefig(plot_file_path)
    plt.close()
    #print(f"Plot saved to {plot_file_path}")

def train_and_evaluate(params):
    # Initialize the model
    model = HAFR(dataset.num_users, dataset.num_items, dataset.cold_num, dataset.num_ingredients, dataset.image_size, params).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # Optimizer and loss function
    #optimizer = optim.Adagrad(model.parameters(), lr=params['lr'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss since model outputs logits

    # Start training
    train_losses, train_accuracies, val_losses, val_accuracies, best_model = train(model, train_dataloader, validation_dataloader, optimizer, criterion, params['epochs'], device)

    # Evaluate on test set
    test_loss, test_accuracy = test(best_model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Ensure the save folder exists
    if not os.path.exists(params['save_folder']):
        os.makedirs(params['save_folder'])

    # Save the model
    save_path = os.path.join(params['save_folder'], f"best_model_{params['try']}.pth")
    torch.save(best_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save the losses
    losses_file_path = os.path.join(params['save_folder'], f"losses_per_epoch_{params['try']}.npy")
    np.save(losses_file_path, {'train_losses': train_losses, 'val_losses': val_losses, "train_accuracies": train_accuracies, "val_accuracies": val_accuracies}, allow_pickle=True)
    print(f"Losses per epoch saved to {losses_file_path}")

    #plot the training curve
    save_training_curve(train_losses, train_accuracies, val_losses, val_accuracies)

    return test_loss, test_accuracy

def print_params(params):
        print(f"batch_size: {params['batch_size']}, "
                f"epochs: {params['epochs']}, "
                f"embed_size: {params['embed_size']}, "
                f"reg: {params['reg']:.5f}, "
                f"reg_image: {params['reg_image']:.5f}, "
                f"reg_w: {params['reg_w']:.5f}, "
                f"reg_h: {params['reg_h']:.5f}, "
                f"lr: {params['lr']:.5f}, "
                f"weight_size: {params['weight_size']}")

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Import the Dataset class
    #from Dataset import Dataset
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    # Initialize dataset
    #dataset = Dataset(params['path'] + params['dataset'])
    dataset = Dataset()#('/content/drive/MyDrive/APS360_Food/Datasets/Data/data')
    print("Finished processing dataset")

    # DataLoaders
    
    Subset_size = 5000

    validation_dataset = CustomDataset(dataset, subset_size=Subset_size, dns=0,  data_type='validation')

    train_dataset = CustomDataset(dataset, subset_size=Subset_size, dns=1, data_type='train')#model.dns

    test_dataset = CustomDataset(dataset, subset_size=Subset_size, dns=0,  data_type='test')
    
    # Verify datasets are disjoint
    train_set = set(zip(train_dataset.user_input, train_dataset.item_input_pos))
    val_set = set(zip(validation_dataset.user_input, validation_dataset.item_input_pos))
    test_set = set(zip(test_dataset.user_input, test_dataset.item_input_pos))

    print("Overlap between training and validation sets:", len(train_set & val_set))
    print("Overlap between training and test sets:", len(train_set & test_set))
    print("Overlap between validation and test sets:", len(val_set & test_set))

    best_params = []
    best_accuracy = 0.0

    for i in range(13, 14):
        params = get_random_hyperparameters(Try_time = i)
        print(f"Hyperparameters try {i}:")
        print_params(params)

        test_loss, test_accuracy = train_and_evaluate(params)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = params

    print(f"\nBest hyperparameters in try {best_params['try']}:")
    print_params(best_params)
    print(f"Best test accuracy: {best_accuracy}")
    
  
