# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


N = 21146
start = 0
file_name = "embeddings_0.txt"


################ Import Packages ################
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#################################################



################# Import Models #################
sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
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



############## Get Close Sentences ##############
def get_close_similarities(claim, justifications, top_n=5):
    # Compute embeddings
    # Compute embeddings in 384d
    claim_embedding = sim_model.encode(claim, convert_to_tensor=True)
    justification_embeddings = sim_model.encode(justifications, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(claim_embedding, justification_embeddings)
    
    # Get indices of top N similarities
    top_n_indices = np.argsort(similarities.numpy(), axis=1)[0, -top_n:][::-1]
    
    # Sort indices by similarity in descending order
    top_n_indices = sorted(top_n_indices, key=lambda idx: similarities[0, idx], reverse=True)
    
    # Get the top N similarity scores
    top_n_similarities = [similarities[0, idx].item() for idx in top_n_indices]
    
    return top_n_indices, top_n_similarities
#################################################



################# Map Source ####################
def map_source(source):
  source = source.lower()
  
  if source == "speech":
    return 0
  
  elif source == "television":
    return 1
  
  elif source == "news":
    return 2
  
  elif source == "blog":
    return 3
  
  elif source == "social_media":
    return 4
  
  elif source == "advertisement":
    return 5
  
  elif source == "campaign":
    return 6
  
  elif source == "meeting":
    return 7
  
  elif source == "radio":
    return 8
  
  elif source == "email":
    return 9
  
  elif source == "testimony":
    return 10
  
  elif source == "statement":
    return 11
  
  elif source == "other":
    return 12
  
  else:
    print("unknown source detected")
    return 13
#################################################


################# Right Wrong ###################
def get_nlis(index, indices, claim, justifications):
  nli_scores = []
    
  for indx in indices:
    # Encode the sentences for the NLI model
    nli_input = nli_tokenizer.encode_plus(claim, justifications[indx], return_tensors='pt', truncation=True)
    
    # Perform NLI prediction
    logits = nli_model(**nli_input).logits
    
    # Get prediction scores
    nli_scores.append(logits[0][2].item())
    nli_scores.append(logits[0][0].item())
    nli_scores.append(logits[0][1].item())
    
  nli_scores.append(map_source(dataset.iloc[index, 4]))

  return nli_scores
#################################################



################ Create Vector ##################
def create_vector(index):
  # extract sentences
  sentences = get_sentences(index)
  
  # get 5 close indices and their similarities
  indices, similarities = get_close_similarities(sentences[0], sentences[1:])
  
  # NLI prediction
  nli_scores = get_nlis(index, indices, sentences[0], sentences[1:])
  
  return np.append(similarities, nli_scores)
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