"""
Module:


"""
from PIL import Image
import numpy as np
import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import json
from dataset import Dataset, safe_collatefn


input_dir = 'malle_dataset/original_images/'
query_dir = 'malle_dataset/modified_images/'
os.makedirs('embeddings', exist_ok=True)
embed_path = 'embeddings/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loadin pretrained resnet50
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)#weights=ResNet50_Weights.IMAGENET1K_V1) # model for classification
resnet50_feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(device) # model to genenrate embeddings ( removed final classification layer)

resnet50_feature_extractor.eval()

# Image preprocessing, transform maintains resnet channel order (C, H, W)
transform = transforms.Compose([
#the next 2 steps preserve aspect ratio, doesnt distort image, and matches imagenet training
transforms.ToImage(),
transforms.ToDtype(torch.float32, scale=True),
transforms.Resize(256),
transforms.CenterCrop(224),

transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225])
])


def embed_batch(batch: torch.tensor):
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

    return batch_embeds

def load_embeddings(filename: str) -> np.ndarray:
    """Loads embeddings from a binary file.

    Args:
        filename (str): The path to the file (without extension).

    Returns:
        np.ndarray: The numerical array loaded from the file.

    Notes:
        The function automatically appends '.npy' to the filename.
    """
    return np.load(filename) # need to fix the way i save this

def save_embeddings(filename:str, embeddings:np.ndarray):
    """Saves embeddings to a binary file.

    Args:
        filename (str): The path where the file will be saved.
        embeddings (np.ndarray): The numerical array to store.

    Notes:
        filename shouldn't include extension
    Returns:
       None
    """
    np.save(f'embeddings/{filename}.npy', embeddings) # need to fix the way i change this

#input_dir should be the folder that contains pics to be embedded
# query is a flag the indictes what type of image we are embedding.
def embed_folder(input_dir, outfile, batch_size, embed_engine: str ):
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


    ''' load and initialize model here'''


    Malle_Dataset = Dataset(input_dir, transform = transform)

    dataLoader = DataLoader(dataset = Malle_Dataset, batch_size = batch_size, collate_fn = safe_collatefn)

    embeddings = []

    for batch, _ in dataLoader:
       batch_embeddings = embed_batch(batch)

       embeddings.append(batch_embeddings)


    embeddings = torch.cat(embeddings, dim=0) # concatenating batched tensors into one  big tensor (image_count, embeddings, 1, 1)
    embeddings = embeddings.squeeze() #  remove of dimensions of size 1, output :(image_count, embeddings)

    normalized_embeddings = torch.nn.functional.normalize(embeddings, p = 2, dim = 1) # dim = 1 means collapse accross the columns, makes sure we ompute unit vectors of embeddings

    normalized_embeddings = normalized_embeddings.cpu().numpy()

    save_embeddings(filename=f'{embed_engine}_{outfile}', embeddings=normalized_embeddings) # need to change how i save this file

def stream_jsonl(filename: str):
    """Yields one JSON object at a time from a .jsonl file."""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def load_jsonl(filename: str):
    """Loads data from a .jsonl file.

    Args:
        filename: Path to the .jsonl file.

    Returns:
        list: A list of dictionaries.
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_query_metadata(query_input_dir: str, query_outfile: str) -> dict:
    """
    Function: extract_query_metadata(input_dir: str, outfile: str) -> None
    ---
    Purpose: extracts query (modified) images metadata from a folder and upload them in a jsonl file
    ---
    Params:
        query_input_dir: path to folder that stores images
        query_outfile: path to folder that stores image metadata
    ---
    Returns: None
    ---
    Notes:
        each line of the jsonl file is a json object that contains image metadata
        query metadata object: {'id':idx, 'class': img_class, 'instance_id': img_name_list[1], 'modifications': mods, 'path': image_path  }
    """
    with open(query_outfile, 'a', encoding= 'utf-8') as f:
        idx = 0 # used to assign unique id to images in a folder

        for img_name in os.listdir(query_input_dir):

            img_name_list = img_name.split('_') # brake downn img name

            # extract img class from broken down img name
            img_class = img_name_list[0]

            # contrust img path with the housing directory and image name
            image_path = os.path.join(query_input_dir, img_name)

            mods = img_name_list[2:]

            # separate last mod from extension
            mods[-1] = mods[-1].split('.')[0]

            f.write (
                json.dumps({'id':idx, 'class': img_class, 'instance_id': img_name_list[1], 'modifications': mods, 'path': image_path  }) + '\n'
                )

            idx +=  1



def extract_index_metadata(index_input_dir: str, index_outfile: str) -> dict:
    """
    Function: extract_metadata(input_dir: str, outfile: str) -> None
    ---
    Purpose: extracts index (original) images metadata from a folder containing images and upload them in a jsonl file
    ---
    Params:
        input_dir: path to folder that stores images
        outfile: path to folder that stores image metadata
    ---
    Returns: None
    ---
    Notes:
        each line of the jsonl file is a json object that contains image metadata
        index metadata object: {'id':idx, 'class': img_class, 'instance_id': instance_id, 'path': image_path}
    """
    with open(index_outfile, 'a', encoding='utf-8') as f:
        idx = 0 # used to assign unique id to images in a folder

        for img_name in os.listdir(index_input_dir):

            img_name_list = img_name.split('_') # brake downn img name

            # extract img class from broken down img name
            img_class = img_name_list[0]

            # contruct img path with the housing directory and image name
            image_path = os.path.join(index_input_dir, img_name)

            # after splitting by '_' orginal images would have instance_id joined with extension (e.g. 2291.JPEG), this line extracts instance id
            instance_id, _ = (img_name_list[-1]).rsplit('.',1)

            f.write(
                json.dumps( {'id':idx, 'class': img_class, 'instance_id': instance_id, 'path': image_path} ) + '\n'
                )

            idx +=  1





if __name__ == '__main__':
    # generate embeddings for orginal images (indexes)
    embed_folder(input_dir= input_dir,
                outfile = 'index',
                batch_size=16,
                embed_engine='resnet50')

    ## generate embeddings for modified copies (queries)
    # embed_folder(input_dir= query_dir,
    #             outfile = 'query',
    #             batch_size=16,
    #             embed_engine='resnet50')


    # # meta_data generation
    # #queryies
    # extract_query_metadata(query_input_dir=query_dir, query_outfile='metadata/queries_metadata.jsonl')

    # #index
    # extract_index_metadata(index_input_dir=input_dir, index_outfile='metadata/index_metadata.jsonl')





