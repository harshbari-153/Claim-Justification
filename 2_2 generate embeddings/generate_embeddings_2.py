# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


N = 21146
start = 4000
file_name = "embeddings_1.txt"


################ Import Packages ################
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#################################################



################# Import Models #################
sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
#nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
#################################################



################# Get Sentence ##################
def get_sentences(index):
  sentences = []
  
  temp = dataset.iloc[index, 2]
  sentences.append(temp)
  
  path = "../1 extract justification/justification/j_" + str(index) + ".txt"
  
  with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    
    for line in file:
      sentences.append(line)
      
  return sentences
#################################################


def get_average(justification_embeddings, top_n_indices):
    # Extract embeddings corresponding to the indices in top_n_indices
    selected_embeddings = justification_embeddings[top_n_indices]
    
    # Calculate the average of these embeddings
    average_embedding = torch.mean(selected_embeddings, dim=0)
    
    # Reshape the result to be (1 x k)
    return average_embedding.unsqueeze(0)



############## Get Close Sentences ##############
def get_embeddings(claim, justifications, top_n=5):
    # Compute embeddings in 384d
    claim_embedding = sim_model.encode(claim, convert_to_tensor=True)
    justification_embeddings = sim_model.encode(justifications, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(claim_embedding, justification_embeddings)
    
    # Get indices of top N similarities
    top_n_indices = np.argsort(similarities.numpy(), axis=1)[0, -top_n:][::-1]
    
    # Sort indices by similarity in descending order
    top_n_indices = sorted(top_n_indices, key=lambda idx: similarities[0, idx], reverse=True)
    
    average_justification_embeddings = get_average(justification_embeddings, top_n_indices)
    
    return np.append(claim_embedding, average_justification_embeddings)
#################################################




################ Create Vector ##################
def create_vector(index):
  # extract sentences
  sentences = get_sentences(index)
  
  return get_embeddings(sentences[0], sentences[1:], 5)
#################################################



dataset = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)
file = open(file_name, "w", encoding='utf-8')

i = 0
while i < 4000:
  vector = create_vector(i+start)
  vector_string = ' '.join(map(str, vector))
  file.writelines(vector_string)
  file.writelines("\n")
  
  i += 1
  # show progress
  if i % 10 == 0:
    print(str(i/40) + "% done")


file.close()

# sucess
print("Vectorization sucessfully done")