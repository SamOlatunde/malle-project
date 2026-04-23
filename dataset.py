"""
Module: 


"""
from typing import List, Any
import os
import torchvision.io
import torch 


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
        """Retrieves an image and its label by index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            tuple: A tuple containing (img, label) if successful, 
                or None if the image could not be loaded.
        """
        image_path = os.path.join(self.input_dir, self.labels[idx])
        
        try:
            img = torchvision.io.decode_image(image_path, mode='RGB')
        except Exception as e:
            print(f'Error loading {image_path}: {e}')
            return None 
        
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label 
    

def safe_collatefn(batch: List[Any]):
    """Filters out None samples from a batch during loading.

    Args:
        batch: A list of samples (images/labels) retrieved by the Dataset.

    Returns:
        A collated batch of tensors, or an empty list if all samples were None.
    """
    batch = [x for x in batch if x is not None]

    if not batch:
        print('Batch is empty - all samples were corrupted.')
        return []
    
    return torch.utils.data.default_collate(batch)