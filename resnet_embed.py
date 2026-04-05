"""
Module: 


"""
from PIL import Image
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('embed_index_result/embeds', exist_ok=True)

embed_path = 'embed_index_result/embeds/'

#loadin pretrained resnet50
resnet50 = models.resnet50(pretrained=True)#weights=ResNet50_Weights.IMAGENET1K_V1) # model for classification
resnet50_feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(device) # model to genenrate embeddings ( removed final classification layer)

resnet50_feature_extractor.eval()

# Image preprocessing, transform maintains resnet channel order (C, H, W)
transform = transforms.Compose([
    #the next 2 steps preserve aspect ratio, doesnt distort image, and matches imagenet training
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])



def embed_image(image_path):
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
    img = Image.open(image_path).convert('RGB') # 
    x = transform(img).unsqueeze(0).to(device) # transform the image, add batch dimensions, and move to device model is on

    with torch.no_grad():
        embed = resnet50_feature_extractor(x) # embed (1,2048,1,1)
    
    #remove batch dimension, move to cpu,  convert to numpy because faiss only takes numpy
    embed = embed.squeeze().cpu().numpy() #shape (2048,)

    # normalize ( we add 1e-10 to avoid dividing by zero in the rare situation were embed is a zero vector) 
    #normalizing ensres magnitide is 1 which makes faiss indexflatip search focus on the direction of vectors
    embed = embed / (np.linalg.norm(embed) + 1e-10) 
    return embed


#input_dir should be the folder that contains pics to be embedded 
# query is a flag the indictes what type of image we are embedding. 
def embed_folder(input_dir, is_query, outfile ):
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
    embeddings = []

    for img_name in os.listdir(input_dir):
        # contrust img path with the housing directory and image name 
        image_path = os.path.join(input_dir, img_name)

        # attempt to genenrate embedding ad store in vec, if it doesnt work, catch the expection and inform the user 
        try:
            vec = embed_image(image_path)
        except Exception as e:
            print('error', image_path, e)
            continue

        embeddings.append(vec.astype( 'float32' )) #we convert to float 32 because faiss expects float32 and numpy default is float64


    # right now, embedding is a list of vectors, vstack converts it to a 2d numpy array (n,2048)
    embeddings = np.vstack(embeddings)

    with open(outfile, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'meta_data': meta_data}, f)


def extract_query_metadata(input_dir: str, img_name: str, idx: int) -> dict:
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
     query metadata object: {'id':idx, 'class': img_class, 'instance_id': img_name_list[1], 'modifications': mods, 'path': image_path  }
    """
    img_name_list = img_name.split('_') # brake downn img name 
    
    # extract img class from broken down img name
    img_class = img_name_list[0]

    # contrust img path with the housing directory and image name 
    image_path = os.path.join(input_dir, img_name)

    mods = img_name_list[2:]

    # separate last mod from extension
    mods[-1] = mods[-1].split('.')[0]
    
    return {'id':idx, 'class': img_class, 'instance_id': img_name_list[1], 'modifications': mods, 'path': image_path  }

   
def extract_index_metadata(input_dir: str, img_name: str, idx: int) -> dict:
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
     index metadata object: {'id':idx, 'class': img_class, 'instance_id': instance_id, 'path': image_path}
    """
    img_name_list = img_name.split('_') # brake downn img name 
                
    # extract img class from broken down img name
    img_class = img_name_list[0]

    # contrust img path with the housing directory and image name 
    image_path = os.path.join(input_dir, img_name)

    # ater splitting by '_' orginal images would have instance_id joined with extension (e.g. 2291.JPEG), this line extracts instance id 
    instance_id, _ = (img_name_list[-1]).rsplit('.',1)

    return {'id':idx, 'class': img_class, 'instance_id': instance_id, 'path': image_path}


def extract_metadata(input_dir: str, is_query: bool, outfile: str) -> None:
    """
    Function: extract_metadata(input_dir: str, is_query: bool, outfile: str) -> None
    ---
    Purpose: extracts image metadata from a folder containing images and upload them in a jsonl file
    ---
    Params:
        input_dir: path to folder that stores images
        is_query: are the images queries or indices in relation to similarity search ( True -> Yes )
        outfile: path to folder that stores image metadata 
    ---
    Returns: None
    ---
    Notes: each line of the jsonl file is a json object that contains image metadata
    """
    with open(outfile, 'a') as f:
        idx = 0 # used to assign unique id to images in a folder  
        
        #If query = True, we are embedding a modified image(query image), 
        #if false we are embedding an orginial image
        if is_query:
            for img_name in os.listdir(input_dir):

                query_metadata = extract_query_metadata (input_dir, img_name , idx)
                f.write (json.dumps(query_metadata) + '\n')
                idx +=  1
        else:
            for img_name in os.listdir(input_dir):

                index_metadata = extract_index_metadata (input_dir, img_name, idx)
                f.write(json.dumps(index_metadata) + '\n')
                idx +=  1



if __name__ == '__main__':
    # generate embeddings for orginal images (indexes)
    embed_folder(input_dir = 'malle_dataset/original_images/',
                 is_query = False,
                 outfile = f'{embed_path}/index_resnet50_embeddings.pkl')
    
    # generate embeddings for modified copies (queries)
    embed_folder(input_dir = 'malle_dataset/modified_images/',
                 is_query = True,
                 outfile = f'{embed_path}/queries_resnet50_embeddings.pkl')
  
        


    
