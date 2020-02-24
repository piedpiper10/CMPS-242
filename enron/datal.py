import os
import re
import string
import math
directory='enron'
target_names=['HAM','SPAM']

def get_data(directory):
	sfolder=['enron%d' %i for i in range(1,6)]
	data=[]
	target=[]
	
	for subfolder in sfolder:
	#loading spam data
		spam_files=os.listdir(os.path.join(directory,sfolder,'spam'))
		for spam_file in spam_files:
			data.append(f.read)
			target.append(1)
	#loading ham data
		ham_files=os.listdir(os.path.join(directory,sfolder,'ham'))
                for ham_file in ham_files:
                        data.append(f.read)
                        target.append(0)
	return data,target
def get_data1(directory):
        sfolder=['enron%d' %i for i in range(6,7)]
        data=[]
        target=[]

        for subfolder in sfolder:
        #loading spam data
                spam_files=os.listdir(os.path.join(directory,sfolder,'spam'))
                for spam_file in spam_files:
                        data.append(f.read)
                        target.append(1)
        #loading ham data
                ham_files=os.listdir(os.path.join(directory,sfolder,'ham'))
                for ham_file in ham_files:
                        data.append(f.read)
                        target.append(0)
        return data,target

def clean(s):
	translator=str.maketrans(""," ", string.punctuation)
	return s.translate(translator)

def tokenize(text):
	text=clean(text).lower()
	return re.split("\W+",text)

def wordcount(words):
	wordcount=[]
	for word in words:
		word_count[word]=wordcount.get(word,0.0+1.0)
	return word_counts

def fit(self, X, Y):
    self.num_messages = {}
    self.log_class_priors = {}
    self.word_counts = {}
    self.vocab = set()
 
    n = len(X)
    self.num_messages['spam'] = sum(1 for label in Y if label == 1)
    self.num_messages['ham'] = sum(1 for label in Y if label == 0)
    self.log_class_priors['spam'] = math.log(self.num_messages['spam'] / n)
    self.log_class_priors['ham'] = math.log(self.num_messages['ham'] / n)
    self.word_counts['spam'] = {}
    self.word_counts['ham'] = {}
 
    for x, y in zip(X, Y):
        c = 'spam' if y == 1 else 'ham'
        counts = self.wordcount(self.tokenize(x))
        for word, count in counts.items():
            if word not in self.vocab:
                self.vocab.add(word)
            if word not in self.word_counts[c]:
                self.word_counts[c][word] = 0.0
 
            self.word_counts[c][word] += count
def predict(self, X):
    result = []
    for x in X:
        counts = self.wordcount(self.tokenize(x))
        spam_score = 0
        ham_score = 0
        for word, _ in counts.items():
            if word not in self.vocab: continue
            
            # add Laplace smoothing
            log_w_given_spam = math.log( (self.word_counts['spam'].get(word, 0.0) + 1) / (self.num_messages['spam'] + len(self.vocab)) )
            log_w_given_ham = math.log( (self.word_counts['ham'].get(word, 0.0) + 1) / (self.num_messages['ham'] + len(self.vocab)) )
 
            spam_score += log_w_given_spam
            ham_score += log_w_given_ham
 
        spam_score += self.log_class_priors['spam']
        ham_score += self.log_class_priors['ham']
 
        if spam_score > ham_score:
            result.append(1)
        else:
            result.append(0)
    return result

X, y = get_data(directory)

MNB = SpamDetector()
MNB.fit(X,y)
x1,y1=get_data1(directory)
pred=MNB.predict(x1)
accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
print("{0:.4f}".format(accuracy))
