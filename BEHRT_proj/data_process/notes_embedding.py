import os
import demjson
import json
import torch.multiprocessing as multiprocessing
from sentence_transformers import SentenceTransformer

# Set GPU and load SentenceTransformer model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_id = 'BAAI/bge-base-en'
model = SentenceTransformer(model_id).to('cuda')

# Define directories
directory = "/data/datasets/Aokun/2023_retreat/n2c2_chunks_revised/"
write_directory = "/data/datasets/lvmengxian/23_10_Long_Documents/"

#process chunks in notes
def process_file(filename):
    try:
        label = filename.split("_")[0]
        file_id = filename.replace(".json","")
        
        #load file and get texts from chunks 
        with open(os.path.join(directory, filename)) as f:
            data = demjson.decode(f.read())
            texts = list(data[label].values())
            
            #run model embedding
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            #modify data structure to fit elasticsearch
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = {"id": f"{file_id}_{i}", "text": texts[i], "text_embedding": {"predicted_value": embedding.tolist(), "model_id": model_id }}
                results.append(result)
            #store  json file
            with open(os.path.join(write_directory, filename),"w") as output:
                for result in results:
                    json.dump(result, output)
                    output.write("\n")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]
    with multiprocessing.Pool() as pool:
        pool.map(process_file, files)
