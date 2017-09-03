from flask import Flask, render_template, request
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

fetch_20newsgroups()
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect.vocabulary_.get(u'algorithm')
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
with open('trained_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def test_new_article(txt):
    X_new_counts = count_vect.transform([txt])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    preds = clf.predict(X_new_tfidf)
    return [twenty_train.target_names[i] for i in preds]

@app.route('/hello/<user>')
def hello_name(user):
   return render_template('hello.html', name = user)

@app.route('/entry')
def enrty_form():
    return render_template('entry.html')

@app.route('/')
def my_form():
    return render_template("entry.html")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    return test_new_article(text)[0]

if __name__ == '__main__':
   app.run(debug = True)
