from flask import render_template, request, jsonify
from app import app
import os
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
    return text.split(' ')

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
	text = request.form.get('text', None)
	assert text is not None

	lr_tfidf = Pipeline([(
		'vect', TfidfVectorizer(analyzer='word', binary=False,
								decode_error='strict', encoding='utf-8',
                                input='content', lowercase=False, max_df=1.0,
                                max_features=None, min_df=1,
                                ngram_range=(1, 1), norm='l2',
                                preprocessor=preprocessor,
                                smooth_idf=True, stop_words=None,
                                strip_accents=None,
                                token_pattern='(?u)\\b\\w\\w+\\b',
                                tokenizer=tokenizer,
                                use_idf=True, vocabulary=None)),
	('clf', LogisticRegression(C=1.0, class_weight=None, dual=False,
								fit_intercept=True, intercept_scaling=1,
								l1_ratio=None, max_iter=100,
                                multi_class='ovr', n_jobs=-1, penalty='l2',
                                random_state=0, solver='liblinear',
                                tol=0.0001, verbose=0, warm_start=False))],
         verbose=None)

	train = pd.read_csv('train.csv', encoding='latin-1')

	X = train['SentimentText']
	y = train['Sentiment']

	lr_tfidf.fit(X, y)

	#tweet_classifier = pickle.load(open('logisticRegression.pkl', 'rb'))
	prob_neg, prob_pos = lr_tfidf.predict_proba([text])[0]
	s = 'Positive' if prob_pos >= prob_neg else 'Negative'
	p = prob_pos if prob_pos >= prob_neg else prob_neg
	return jsonify({
		'sentiment': s,
		'probability': p
		})