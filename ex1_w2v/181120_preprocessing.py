import numpy as np
import os, re, csv, math, codecs
from keras.models import Sequential
from keras.layers import Embedding
from tqdm import tqdm

np.random.seed(0)

# NUM_WORDS, train으로 input받은 단어의 수
vocab_size = len()+1 
EMB_DIM = 300
embeddings_index = dict()
# INPUT 문장길이
text_max_words = 

##########################################1. 데이터셋 생성하기

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 훈련셋과 검증셋 분리
x_val = x_train[20000:]
y_val = y_train[20000:]
x_train = x_train[:20000]
y_train = y_train[:20000]
# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)



#load embeddings
print('loading word embeddings...')
f = codecs.open('파일이름.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, EMB_DIM))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


#####################################2. 모델 구성하기
# input_length : 단어의 수 즉 문장의 길이를 나타냅니다
# 임베딩 레이어의 출력 크기는 샘플 수 * output_dim * input_lenth가 됩니다
emb = Embedding(input_dim=vocab_size, output_dim=EMB_DIM,
                trainable=False, weights=[embedding_matrix], input_length=text_max_words)

# 임베딩 dropout

# GaussianNoise(mean=0.0, stddev=0.2)
keras.layers.GaussianNoise(stddev=0.2)
keras.layers.GaussianDropout(rate)

# LSTM(310, 250, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
# Dropout(p=0.3)
model.add(Bidirectional(LSTM(250, return_sequences=True), input_shape=(n_timesteps, 1)))

#         SelfAttention(
#             Sequential(
#                 Linear(in_features=500, out_features=1, bias=True)
#                 Tahh()
#                 Dropout(0.3)
#                 Linear(in_features=500, out_features=1, bias=True)
#                 Tanh()
#                 Dropout(p=0.3)
#             )
#             Softmax()
#         )
#     )
#     Linear(in_features=500, out_features=11, bias=True)    binary classification
# )