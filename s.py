'''
Module: s.py
---
For fast pototyping experiemnets 
'''
from typing import Any
import os
import numpy as np
import torchvision.transforms.v2 as transforms
import torchvision.io
from torchvision.models import resnet50, ResNet50_Weights

import torch 
from torch.utils.data import DataLoader




np.set_printoptions(precision=4, suppress=True)

input_dir = 'malle_dataset/original_images/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loadin pretrained resnet50
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)#weights=ResNet50_Weights.IMAGENET1K_V1) # model for classification
resnet50_feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(device) # model to genenrate embeddings ( removed final classification layer)

resnet50_feature_extractor.eval()


def embed_batch(batch):
    """ 
    Function:
    ---
    Purpose:
    ---
    Params:
    ---
    Returns:
    ---
    Notes:
   
    """
    batch = batch.to(device) #move batch to device model is on
    
    with torch.no_grad():
        batch_embeds = resnet50_feature_extractor(batch) # embed (1,2048,1,1)
    
    #remove batch dimension, move to cpu,  convert to numpy because faiss only takes numpy
    #embed = embed.squeeze().cpu().numpy() #shape (2048,)

    # normalize ( we add 1e-10 to avoid dividing by zero in the rare situation were embed is a zero vector) 
    #normalizing ensres magnitide is 1 which makes faiss indexflatip search focus on the direction of vectors
    
    #embed = embed / (np.linalg.norm(embed) + 1e-10) 
    return batch_embeds

class Dataset(torch.utils.data.Dataset):
    '''
    Class: Malle_Dataset
    ---
    Purpose:
    ---
    Member Variables:
        self.input_dir: directory containing the dataset
        self.transform: transform to be applied each image in the dataset
        self.target_transform: transform to be applied to labels 
        self.labels: the file names, we can deduce information 
    '''
    def __init__(self, input_dir: str, transform = None, target_transform = None) -> None:
        '''
        Member Function: __init__(self, input_dir: str, transform = None, target_transform = None) -> None
        ---
        Params: 
            input_dir: directory containing the dataset
            transform: transform to be applied each image in the dataset
            target_transform: transform to be applied to labels 
        ---
        Purpose:
        ---
        Returns:
        ---
        Notes:
        '''
        self.input_dir = input_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels = os.listdir(input_dir)
    
    def __len__(self) -> int:
        '''
        Member Function:__len__(self) -> int
        ---
        Params: 
        ---
        Purpose:
        ---
        Returns:
        ---
        Notes:
        '''
        return len(self.labels)

    def __getitem__(self, idx: int) -> Any:
        '''
        Member Function: __getitem__(self, idx: int) -> Any
        ---
        Params: 
        ---
        Purpose:
        ---
        Notes:
        '''
        image_path = os.path.join(self.input_dir, self.labels[idx])
        
        img = torchvision.io.decode_image(image_path, mode='RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label 


def save_embeddings(filename:str, embeddings:np.ndarray):
    """Saves embeddings to a binary file.

    Args:
        filename (str): The path where the file will be saved.
        embeddings (np.ndarray): The numerical array to store.

    Returns:
       None
    """
    np.save(filename, embeddings)
 

if __name__ == '__main__':
    transform = transforms.Compose([
    #the next 2 steps preserve aspect ratio, doesnt distort image, and matches imagenet training
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
    ])
    
    Malle_Dataset = Dataset(transform = transform,
        input_dir='malle_dataset/original_images/',
        )
    
    batch_size = 16

    DataLoader = DataLoader(dataset = Malle_Dataset, batch_size = batch_size)


    # ''' verifying sameness:  converting to numpy then finding unit vectors VS finindg unit then converting to numpy''' 
    # one, _ = next(iter(DataLoader))
  
    # np_first = embed_batch(one)
    # torch_first = embed_batch(one)
    
    # ''' np first routine '''
    # np_first = np_first.squeeze().cpu().numpy() #shape (2048,)
    # np_first = np_first / (np.linalg.norm(np_first) + 1e-10) 
    
    # ''' torch_first routine '''
    # torch_first = torch_first.squeeze() #  remove of dimensions of size 1, output :(image_count, embeddings)
    # torch_first = torch.nn.functional.normalize(torch_first, p = 2, dim = 0) # dim = 1 means collapse accross the columns, makes sure we ompute unit vectors of embeddings
    # torch_first = torch_first.cpu().numpy()

    
    # print(np_first.shape)
    # print(torch_first.shape)
    
    # print('All Close: ', np.allclose(np_first,torch_first))
    # print(f'np Is zero: { np.array_equiv(np_first, 0)} \t', f' torch Is zero: { np.array_equiv(torch_first, 0)}')
   
    # print('\n\n\n')

    # print('np_first: ', np_first[85:95], end='\n\n\n')
    # print('torch_first: ', torch_first[19:32], end='\n\n\n')
     

    # print("First ", np_first.max())
    # print("Dot ", np.dot(np_first, torch_first))

    embeddings = []

    for batch, _ in DataLoader: #i in range(0,len(os.listdir('malle_dataset/original_images/')), batch_size):
        # batch, _ = next(iter(DataLoader))    
        # print(" Batch Type:", type(batch))
        # print (" Batch Shape:", batch.shape, end='\n\n\n')
         # print(" Batch Type:", type(batch))
        # print (" Batch Shape:", batch.shape, end='\n\n\n')
        batch_embeddings = embed_batch(batch)

        embeddings.append(batch_embeddings)
    

    embeddings = torch.cat(embeddings, dim=0) # concatenating batched tensors into one  big tensor (image_count, embeddings, 1, 1)
    embeddings = embeddings.squeeze() #  remove of dimensions of size 1, output :(image_count, embeddings)

    normalized_embeddings = torch.nn.functional.normalize(embeddings, p = 2, dim = 1) # dim = 1 means collapse accross the columns, makes sure we ompute unit vectors of embeddings
    
    print(normalized_embeddings.dtype)
    normalized_embeddings = normalized_embeddings.cpu().numpy()
    print(normalized_embeddings.dtype)
    save_embeddings(filename='embeddings/resnet50_index.npy', embeddings=normalized_embeddings)

    a = np.load('embeddings/resnet50_index.npy')

    print(a.shape)
    
    

       








'''
#experimenting with clip
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints:  [[0.9927937  0.00421068 0.00299572]]

'''
