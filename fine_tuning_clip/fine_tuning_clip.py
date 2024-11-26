import clip
import torch
import json
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from transformers import CLIPProcessor, CLIPModel
import requests
from transformers import CLIPModel
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
batch_size = 32

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    device_map=device,
    torch_dtype=torch.float32,
)
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def custom_collate_fn(batch):
    # print("111111111111111111111")
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'].squeeze(0) for item in batch], batch_first=True)
    # print("222222222222222222222")
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'].squeeze(0) for item in batch], batch_first=True)
    # print("333333333333333333333")
    pixel_values = torch.stack([item['pixel_values'].squeeze(0) for item in batch])
    # print("444444444444444444444")
    # input_ids = input_ids.view(input_ids.size(0), -1)
    # attention_mask = attention_mask.view(attention_mask.size(0), -1)
    #pixel_values = pixel_values.view(pixel_values.size(0), -1, 224, 224)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values}


class image_title_dataset():
    def __init__(self, list_image_path, list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = list_txt
    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert("RGB")
        image = image.resize((224, 224))
        #image.show()
        title = self.title[idx]
        title = title[:76].ljust(76)
        inputs = processor(text=[title], images=image, return_tensors="pt", padding=True,truncation=True,max_length=76)
        return inputs
    

def main():
    image_base_path = ".\\data_image\\"
    text_base_path = ".\\data_text\\"
    image_folder_names = [f for f in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, f))]
    text_folder_names = [f for f in os.listdir(text_base_path) if os.path.isdir(os.path.join(text_base_path, f))]
    
    common_folder_names = list(set(image_folder_names).intersection(text_folder_names))

    print(len(common_folder_names))

    
    
    list_text = []
    list_image_path = []
    
    #load the paths of the images and the text
    for folder_name in common_folder_names:
        image_folder_path = os.path.join(image_base_path, folder_name)
        text_folder_path = os.path.join(text_base_path, folder_name)
        
        
        # there's one text description for each foler. But there's multiple images in each folder
        # so we need to create a pair of image and text for each image in the folder
        # and we create dataset based on that
        # then we can create a dataloader for the dataset
        # then we can load the model and fine-tune the model based on our dataloader
        image_path = os.path.join(image_folder_path, image_folder_path)
        
        image_path = os.path.join(image_folder_path, "images")
        # there's many images in the image folder
        
        # there's one text description for each foler. We need to load the description and steps and add them together
        text_path_descriptions = os.path.join(text_folder_path, "descriptions")
        text_path_descriptions = os.path.join(text_path_descriptions, "descriptions.txt")
        text_path_steps = os.path.join(text_folder_path, "steps")
        text_path_steps = os.path.join(text_path_steps, "steps.txt")
        
        # load the text description and steps and add them up together to form a string
        with open(text_path_descriptions, "r", encoding="utf-8") as f:
            description = f.read()
        with open(text_path_steps, "r", encoding="utf-8") as f:
            steps = f.read()
        text = description + steps
        
        for image_file_name in os.listdir(image_path):
            image_file_path = os.path.join(image_path, image_file_name)
            list_image_path.append(image_file_path)
            list_text.append(text)
    
    
    
    # modify the length of the dataset for testing
    # dataset = image_title_dataset(list_image_path[1:200], list_text[1:200])
    dataset = image_title_dataset(list_image_path, list_text)
    print(f"len of dataset: {len(dataset)}")
    print(f"len of list_image_path: {len(list_image_path)}")
    print(f"len of list_text: {len(list_text)}")
                
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    dataset = image_title_dataset(list_image_path, list_text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    train_num = int(len(dataset) * 0.8)
    val_num = int(len(dataset) * 0.1)
    test_num = len(dataset) - train_num - val_num
    
    #split the dataset into training and validation dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # to do: split the dataset into training and validation dataset
    
    
    # print the shape of each element in batch, and input
    # print(f"Number of batches: {len(dataloader)}")
    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i+1}:")
    #     for key, value in batch.items():
    #         print(f"  {key}: {value.shape}")
    #     if i == 2:  # Print only the first 3 batches for brevity
    #         break


    for epoch in range(2):
        model.train()
        # to do: use train_test_split to split the dataset into training and validation dataset
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            inputs = batch
            # print(f"inputs: {inputs}")
            # print(f"inputs.keys: {inputs.keys()}")
            # print(f"input shape: {inputs['input_ids'].shape}")
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)
            inputs['attention_mask'] = inputs['attention_mask'].view(inputs['attention_mask'].size(0), -1).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            # print(f"logits_per_image: {logits_per_image}")
            # print(f"logits_per_text.shape: {logits_per_text.shape}")
            # print(f"logits_per_text: {logits_per_text}")
            
            # print(f"len of logits_per_image: {len(logits_per_image)}")
            ground_truth = torch.arange(batch_size,dtype=torch.long,device=device)
            #print(f"ground_truth: {ground_truth}")
            total_loss = (criterion(logits_per_image,ground_truth) + criterion(logits_per_text,ground_truth))/2
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1} - Loss: {total_loss.item():.4f}")

        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            #to do: add validation dataset
            for batch in val_dataloader:
                inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)
                inputs['attention_mask'] = inputs['attention_mask'].view(inputs['attention_mask'].size(0), -1).to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                ground_truth = torch.arange(len(logits_per_image)).to(device)
                loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2
                val_loss += loss.item()

        val_loss /= len(dataloader)
        print(f"Validation Loss after epoch {epoch}: {val_loss:.4f}")

if __name__ == "__main__":
    main()