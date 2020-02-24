from __future__ import division
import numpy as np
import os
import re
import string
import math
from io import open
from nltk.corpus import stopwords
DATA_DIR = 'enron'
target_names = ['ham', 'spam']

def get_data(DATA_DIR):
	subfolders = ['enron%d' % i for i in range(1,6)]

	data = []
	target = []
	for subfolder in subfolders:
		# spam
		spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
		for spam_file in spam_files:
			with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(1)

		# ham
		ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
		for ham_file in ham_files:
			with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(0)

	return data, target

def get_data1(DATA_DIR):
        subfolders = ['enron%d' % i for i in range(6,7)]

        data = []
        target = []
        for subfolder in subfolders:
                # spam
                spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
                for spam_file in spam_files:
                        with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
                                data.append(f.read())
                                target.append(1)

                # ham
                ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
                for ham_file in ham_files:
                        with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
                                data.append(f.read())
                                target.append(0)

        return data, target

class SpamDetector(object):
    max_word_spam="0"
    max_word_ham="0"
    max_count_ham=0
    max_count_spam=0;
    """Implementation of Naive Bayes for binary classification"""
    def clean(self, s):
	stop = set(stopwords.words('english'))
	querywords = s.split()

	resultwords  = [word for word in querywords if word.lower() not in stop]
	result = ' '.join(resultwords)
	s=result
        for c in string.punctuation:
                s= s.replace(c,"")
        return s

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
		word_counts[word] = word_counts.get(word, 0.0) + 1.0 
	return word_counts

    def fit(self, X, Y):
        """Fit our classifier
        Arguments:
            X {list} -- list of document contents
            y {list} -- correct labels
        """
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
	print type(X)
        n = len(X)
        self.num_messages['spam'] = sum(1 for label in Y if label == 1)
        self.num_messages['ham'] = sum(1 for label in Y if label == 0)
        self.log_class_priors['spam'] = math.log(self.num_messages['spam'] / n)
        self.log_class_priors['ham'] = math.log(self.num_messages['ham'] / n)
        self.word_counts['spam'] = {}
        self.word_counts['ham'] = {}
	
        for x, y in zip(X, Y):
            c = 'spam' if y == 1 else 'ham'
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                
                self.word_counts[c][word] += count
  		if self.word_counts['spam'].get(word, 0.0)>self.max_count_spam and word!='subject':
                        self.max_word_spam=word
                        self.max_count_spam=self.word_counts['spam'].get(word, 0.0)
                if self.word_counts['ham'].get(word, 0.0)>self.max_count_ham and word!='subject':
                        self.max_word_ham=word
                        self.max_count_ham=self.word_counts['spam'].get(word, 0.0)

               
 
	
    def predict(self, X):

	result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            spam_score = 0
            ham_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue
                # add Laplace smoothing
		'''
		if self.word_counts['spam'].get(word, 0.0)>self.max_count_spam and word!='subject':
			self.max_word_spam=word
			self.max_count_spam=self.word_counts['spam'].get(word, 0.0)
		if self.word_counts['ham'].get(word, 0.0)>self.max_count_ham and word!='subject':
                        self.max_word_ham=word
                        self.max_count_ham=self.word_counts['spam'].get(word, 0.0)

		print word,self.word_counts['spam'].get(word,0.0)'''
                log_w_given_spam = math.log( (self.word_counts['spam'].get(word, 0.0) + 1) / (self.num_messages['spam'] + len(self.vocab)) )
                log_w_given_ham = math.log( (self.word_counts['ham'].get(word, 0.0) + 1) / (self.num_messages['ham'] + len(self.vocab)) )

                spam_score += log_w_given_spam
                ham_score += log_w_given_ham

            spam_score += self.log_class_priors['spam']
            ham_score += self.log_class_priors['ham']
    	    print "SPAM:",self.max_word_spam,self.max_count_spam
	    print "HAM:",self.max_word_ham,self.max_count_ham

            if spam_score > ham_score:
                result.append(1)
            else:
                result.append(0)
        return result
        

if __name__ == '__main__':
    X, y = get_data(DATA_DIR)
    MNB = SpamDetector()
    MNB.fit(X, y)
    X1,y1=get_data(DATA_DIR)	
    pred = MNB.predict(X1)
    true = y1
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
    print accuracy
    print("{0:.4f}".format(accuracy))
