import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Video_Games_5.json.gz')
X_train, X_test, y_train, y_test= train_test_split(df.reviewText, df.overall)



clf = MultinomialNB().fit(X_train_tfidf, )





'''
def get_data():
    df = pd.read_json('../data/reviews_Books_5.json')
    return df
'''
