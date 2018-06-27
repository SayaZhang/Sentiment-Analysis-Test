import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

TRAIN_PATH = '../data/aclImdb/train.txt'

# 处理原始数据
def join_data():
	
	pos_location='../data/aclImdb/train/pos'
	pos_files=os.listdir(pos_location)
	neg_location='../data/aclImdb/train/pos'
	neg_files=os.listdir(neg_location)

	all=[]
	for file in pos_files:
	    whole_location=os.path.join(pos_location,file)
	    with open(whole_location,'r',encoding='utf8') as f:
	        line=f.readlines() + ["p"]
	        all.extend([line])
	
	for file in neg_files:
	    whole_location=os.path.join(neg_location,file)
	    with open(whole_location,'r',encoding='utf8') as f:
	        line=f.readlines() + ["n"]
	        all.extend([line])

	df = pd.DataFrame(all,columns=['line','label'])
	df.to_csv(TRAIN_PATH,index=None)

# 加载数据
def load_data(option='all'):
	
	df = pd.read_csv(TRAIN_PATH)
	print("Load Data Success! Length: ",len(df))
	
	# 只加载100条
	if option == 'chunk':
		return df.loc[:100,:]
	
	return df

# 去停用词和标点符号
def delete_stopwords(x):
	
	# load stopwords
	stop_words=set(stopwords.words('english'))
	
	words = nltk.word_tokenize(x)
	line = []
	for word in words:
		if word.isalpha() and word not in stop_words:
			line.append(word)

	return line

def text_preprocessing():

	# load data
	df = load_data()

	# delete stopwords
	df['line'] = df['line'].apply(lambda x: delete_stopwords(x))
	
	# 获取词索引
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(df['line'])
	one_hot_results = tokenizer.texts_to_matrix(df['line'], mode='binary')
	word_index = tokenizer.word_index
	df['word_index'] = tokenizer.texts_to_sequences(df['line'])
	df['word_index_len'] = df['word_index'].apply(lambda x: len(x))

	print(df['word_index_len'].describe())


text_preprocessing()