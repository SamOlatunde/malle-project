'''
Module: s.py
---
For fast pototyping experiemnets 
'''
from typing import Any
import os
import numpy as np
import torchvision.io
from torch import data
input_dir = 'malle_dataset/original_images/'

class Malle_Dataset:
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

    def __getitem__(self, idx: int, trans) -> Any:
        '''
        Member Function: __getitem__(self, idx: int) -> Any
        ---
        Params: 
        ---
        Purpose:
        ---
        Notes:
        '''
        image_path = os.path.join(input_dir, self.labels[idx])
        
        img = torchvision.io.decode(image_path, mode='RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label 










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
