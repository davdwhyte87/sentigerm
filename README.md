# sentigerm
This is a machine learning project, where I built a sentiment analysis program.


# how to set it up
- clone the repo (https://github.com/davdwhyte87/sentigerm)
- cd into the directory `cd sentigerm`
- run `pip install -r requirements.txt` to install requirements
- run `python init.py` to extract data drom the json files in /dutch_data and vectorize them, then serialize them in a pickle file
- run `python main.py` to get the data from the .pkl files and train a model(Random forest classifier) and calculate a score on test data


# issues
- The data is not good, more that half of the datasets are not good

# Things that can be done better
- I could use a better model, like a LSTM Recurrent Neural Network
