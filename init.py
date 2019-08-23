import nltk
import random
import json
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# sentiments are neutral, positive, negative

# this function gets all the data from the files
def load_from_json(file_name):
  data = []
  bad_data = 0
  for file_ in file_name:
    with open("dutch_data/"+file_) as json_file:
      json_data = json.load(json_file)

    for j_data in json_data:
      try:
        if j_data['sentiment'] == 'neutral' or j_data['sentiment'] == 'positive'or j_data['sentiment'] == 'negative':
          data.append({'sentiment':j_data['sentiment'], 'content': j_data['content'].lower()})
      except:
        bad_data = bad_data + 1
  print('Bad data count: ', bad_data)
  return data

raw_data = load_from_json(['dutch1.json', 'dutch2.json', 'dutch3.json', 'dutch4.json'])

# stop words for dutch 
stop_words = list(set(stopwords.words('dutch')))

all_sentences = []
all_text = ' '
for content in raw_data:
  cleaned = re.sub(r'[^(a-zA-Z)\s]','', content['content'])
  all_text = cleaned.lower()+all_text
  all_sentences.append(cleaned)

words = word_tokenize(all_text)

words_freq = nltk.FreqDist(words)

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(all_sentences)
# summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(all_sentences)
# summarize encoded vector
# print(vector.shape)
# print(vector.toarray())

x_data = []
y_data = []
for content in raw_data:
  vector_data = vectorizer.transform([content['content']])
  print(vector_data.toarray()[0])
  x_data.append(vector_data.toarray()[0])
  y = 0
  if content['sentiment'] == "neutral":
    y = 0
  if content['sentiment'] == "positive":
    y = 1
  if content['sentiment'] == "negative":
    y = 2
  y_data.append([y])

# print(x_data[20])
print(len(y_data))
print(len(x_data))
if len(x_data) == len(y_data):
  with open('train_x.pkl','wb') as f:
    pickle.dump(x_data, f)
  f.close()
  with open('train_y.pkl','wb') as fy:
    pickle.dump(y_data, fy)
  fy.close()
else:
  raise Exception("Not complete data")
