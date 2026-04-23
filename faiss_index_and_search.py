import pickle
import faiss
import numpy as np
from embed import load_embeddings, load_jsonl
import json
embed_path = 'embeddings/'
index_path = 'index/'
metadata_path = 'metadata/'
result_path = 'results/'

# load  index embeddings 
index_embeddings = load_embeddings(f'{embed_path}resnet50_index.npy')

# load index metadata
index_metadata = load_jsonl(f'{metadata_path}index_metadata.jsonl')


d = index_embeddings.shape[1]

# build index ( flat index since we aredealing with small dataset)
index = faiss.IndexFlatIP(d) #inner product 
index.add(index_embeddings) # add vectors 

# save index to disk
faiss.write_index(index, f'{index_path}faiss_resnet50_IndexFlatIP.index')

# load queries embedding 
query_embeddings = load_embeddings(f'{embed_path}resnet50_query.npy')
query_metadata = load_jsonl(f'{metadata_path}queries_metadata.jsonl')


k = 3
S_S, I = index.search(query_embeddings, k = k) # S_S: cosine similarity scores, I:indices


# Open the file once, then run your loop inside it
with open(f'{result_path}resnet50_results.jsonl', 'w', encoding='utf-8') as f:
    for i, (s_s, indices) in enumerate(zip(S_S, I)):
        qinfo = query_metadata[i]
        res = []
        
        for score, indx in zip(s_s, indices):
            res.append({
                'score': float(score), 
                'index_id': int(indx), # Ensure JSON serializable
                'index_class': index_metadata[indx]['class'], 
                'index_instance_id': index_metadata[indx]['instance_id'], 
                'index_path': index_metadata[indx]['path'] 
            }) 
        
        # This is the individual result object
        entry = {
            'query_class': qinfo['class'],  
            'query_instance_id': qinfo['instance_id'], 
            'query_path': qinfo['path'], 
            'matches': res
        }
        
        # Write immediately and add a newline
        f.write(json.dumps(entry) + '\n')
