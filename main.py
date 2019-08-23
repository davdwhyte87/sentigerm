import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np 
import random

with open('train_x.pkl','rb') as fr:
  x_data = pickle.load(fr)
fr.close()

all_x_data = []
for x in x_data:
  all_x_data.append(x)

with open('train_y.pkl', 'rb') as fr:
  y_data = pickle.load(fr)
fr.close()

y_data = np.ravel(y_data)

train_x_data = all_x_data
test_x_data = random.sample(list(train_x_data), 100)

train_y_data = y_data
test_y_data = random.sample(list(train_y_data), 100)


# build model 
clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                             random_state=1)
#train
clf.fit(train_x_data, train_y_data)

# pred = clf.predict([test_x_data[20]])
# print(pred)

score = clf.score(test_x_data, test_y_data)
print('model score: ', score)