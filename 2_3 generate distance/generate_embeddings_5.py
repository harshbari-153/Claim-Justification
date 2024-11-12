# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


N = 21146
start = 16000
file_name = "embeddings_4.txt"


################ Import Packages ################
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gensim
from gensim.models import KeyedVectors
import nltk
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



stop_words = ["a", "an", "the", "of", "in", "for", "through", "there", "be", "is", "was", "will", "and", "or", "not", "no", "on", "at", "under", "such", "that", "to", "with"]

# Tokenize
def tokenize(splits):
  return [word for word in splits if word != '' and word not in stop_words]

########## Get distance from sentence ###########
def get_sentence_distance(sentence):
  splits = sentence.split(" ")
  tokens = tokenize(splits)
  pos_tags = nltk.pos_tag(tokens)
  n = len(pos_tags)
  
  subject = False
  action = False
  
  # check word for main subject
  i = 0
  while i < n and subject == False:
    if (pos_tags[i][1][:2] == "NN") and (pos_tags[i][0] in w_model.key_to_index):
      main_subject = pos_tags[i][0]
      subject = True
    i += 1
    
  i = 0
  while i < n and subject == False:
    if (pos_tags[i][1][:2] == "PR" or pos_tags[i][1][-2:] == "DT") and (pos_tags[i][0] in w_model.key_to_index):
      main_subject = pos_tags[i][0]
      subject = True
    i += 1
    
  # check word for action
  i = 0
  while i < n and action == False:
    if (pos_tags[i][1][:2] == "JJ" or pos_tags[i][1][:2] == "VB") and (pos_tags[i][0] in w_model.key_to_index):
      main_action = pos_tags[i][0]
      action = True
    i += 1
  
  if subject == False:
    main_subject = "apple"
    
  if action == False:
    main_action = "apple"
    
  return np.array(w_model[main_subject]) - np.array(w_model[main_action])
#################################################



#### Get Distance between Subject And Action ####
def distance_vector(claim, justifications, top_n_indices):
  # distance between subject and its property in claim
  c_distance = get_sentence_distance(claim)
  
  # distance between subject and its property in justifications
  j_distance = np.array([0 for x in range(100)])
  for i in top_n_indices:
    j_distance = j_distance + get_sentence_distance(justifications[i])
    
  j_distance = j_distance / 5
  
  return c_distance - j_distance
#################################################



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
    
    return distance_vector(claim, justifications, top_n_indices)
#################################################




################ Create Vector ##################
def create_vector(index):
  # extract sentences
  sentences = get_sentences(index)
  
  return get_embeddings(sentences[0], sentences[1:], 5)
#################################################



# open dataset
dataset = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)

# open file to write
file = open(file_name, "w", encoding='utf-8')


try:
  with open('wiki_model.bin', 'r') as f:
    w_model = gensim.models.KeyedVectors.load('wiki_model.bin')
except FileNotFoundError:
  wiki_model = gensim.downloader.load('glove-wiki-gigaword-100')
  wiki_model.save('wiki_model.bin')
  w_model = gensim.models.KeyedVectors.load('wiki_model.bin')



i = 0
while i+start < N:
  vector = create_vector(i+start)
  vector_string = ' '.join(map(str, vector))
  file.writelines(vector_string)
  file.writelines("\n")
  
  i += 1
  # show progress
  if i % 10 == 0:
    print(str(((i+start)*100)/(N)) + "% done")


file.close()

# sucess
print("Vectorization sucessfully done")