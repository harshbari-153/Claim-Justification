# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


################ Import Libraries ###############
import requests
import pandas as pd
from bs4 import BeautifulSoup
import csv
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#################################################


stop_words = stopwords.words('english')
stemmer = PorterStemmer()

############ Statement Preprocessing ############
def preprocess_statement(text):
  if isinstance(text, str):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stop words
    # tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [word for word in tokens if word.isalpha()]
    
    # Apply stemming
    # tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)
#################################################



################ Fetch Justification ############
def fetch_justification(url, sr_no):
  #For chrome users
  headers = {
  "User-Agent": "Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.36",
  "Accept-Language": "en-US,en;q=0.5"}
  
  #For mozilla users
  #headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0'}
  
  #search request
  response = requests.get(url, headers=headers)
  
  #if request accept
  if response.status_code == 200:
    
    #create soup object, get the html structure
    soup = BeautifulSoup(response.text, 'html.parser')
    
    #Create text file
    f = open("justification\j_" + str(sr_no) + ".txt", "w", encoding='utf-8')
    
    
    #fetch title
    title = soup.find('h1', class_="c-title c-title--subline")
    f.write(preprocess_statement(title.text.strip()))
    
    #fetch justification
    article = soup.find('article', class_ = "m-textblock")
    paragraphs = article.find_all('p')
    
    for paragraph in paragraphs:
      para = paragraph.text.split('.')
      
      for line in para:
        string = preprocess_statement(line.strip())
        
        if string != "" and len(string) > 3:
          f.write('\n')
          f.write(string)
    
    #Close file and end sucessfully
    f.close()
    
  #failed to accept request    
  else:
    print("Error Occured: Failed to retrive data at index " + str(sr_no))
#################################################





data = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)

n = len(data)

i = 0
while i < n:
  fetch_justification(data.iloc[i, 7], i)
  i += 1
  
  if i % 50 == 0:
    print(str(i*100/n) + "% done")
    
#Sucess
print("Justification scrapped sucessfully")

# Note: j_598, j_2221, j_3779 is scrapped manually