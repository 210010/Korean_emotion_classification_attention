import numpy as np
import os, re, csv, math, codecs
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer as t
from tqdm import tqdm
from keras.layers import merge, Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import *
# to visualize, and to make zero shape matrix
from attention_utils import get_activations, get_data_recurrent
from Attention import Attention
import pandas as pd
from konlpy.tag import Okt as Twitter
from selfword2vec import tokenization
from Anomaly import checkAnomaly_x_y
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from numpy import argmax
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import binary_crossentropy
import json
# 1. 모델 저장시키기 
# 2. tokenizer와 konlpy morphs 호환 여부 (완성 jupyter note)
# 2'. word2vec 2가지 경우 더 추가 to embedding(final_total word2vec, twitter_translated.vec)
# 3. 변수들 설정하기


np.random.seed(3)

# NUM_WORDS, train으로 input받은 단어의 수
MAX_NB_WORDS = 20000
vocab_size = 0
EMB_DIM = 300
embeddings_index = dict()

# columns = ["ID", "Tweet", "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism","sadness","surprise","trust"]
columns = ["ID","Tweet","분노","기대","혐오스러운","두려움","기쁨","사랑","낙관론","비관론","슬픔","놀라움","믿음"]




##########################################1. 데이터셋 생성하기

train_array = pd.read_csv("/home/minwookje/coding/ex1_w2v/data/tweet/dump/kor_train.txt",sep="\t", header=None,names=columns).values
val_array = pd.read_csv("/home/minwookje/coding/ex1_w2v/data/tweet/dump/kor_dev.txt",sep="\t", header=None,names=columns).values
test_array = pd.read_csv("/home/minwookje/coding/ex1_w2v/data/tweet/dump/kor_test_gold.txt",sep="\t", header=None,names=columns).values

# 판다 shape
print("train_array"+ str(train_array.shape))
print("val_array"+str(val_array.shape))
print("test_array"+str(test_array.shape))

print("Reading data!")
#  x, y 분할하기
x_train = train_array[1:,1]
y_train = train_array[1:,2:]
x_val = val_array[1:,1]
y_val = val_array[1:,2:]
x_test = test_array[1:,1]
y_test = test_array[1:,2:]


print("checking Anomaly!!")
x_train, y_train = checkAnomaly_x_y(x_train,y_train)
x_val, y_val = checkAnomaly_x_y(x_val,y_val)
x_test, y_test = checkAnomaly_x_y(x_test,y_test)

print("ANomaly result!")
print("x_train.shape"+ str(x_train.shape))
print("y_train.shape"+ str(y_train.shape))
print("x_val.shape"+ str(x_val.shape))
print("y_val.shape"+ str(y_val.shape))
print("x_test.shape"+ str(x_test.shape))
print("y_test.shape"+ str(y_test.shape))
# print(type(x_train)) np.array로 변형이 필요한가
print("Finished!")

# jupyter notebook
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

tmp = []
train_tmp = []
val_tmp = []
test_tmp = []
max_count = 0
set_words = set()

print("Tokenizing!")
# test의 tmp만 따로 받아주고 나머지는 val train test모두 통합시켜준다.
test_tmp, dummy , set_words = tokenization(x_test)
val_tmp, dummy , set_words = tokenization(x_val)
train_tmp, max_count , set_words = tokenization(x_train)
tmp, dummy , set_words = tokenization(np.hstack([x_train,x_val,x_test]))
print("Tokenizing finished!")
tmp = [] # tmp없애준다



#token shape
print("It's by len()")
print("train_tmp.len():" + str(len(train_tmp))+","+ str(len(train_tmp[0])))
print("val_tmp shape.len():" + str(len(val_tmp))+","+ str(len(val_tmp[0])))
print("test_tmp shape.len():" + str(len(test_tmp))+","+ str(len(test_tmp[0])))


print("max_count:"+str(max_count))
# 문장길이 100으로 맞춘다.
max_count = min(100, max_count)


print("Readding Embedding file")
# embeddings_index == dict(w2v's word: vector)
f = codecs.open('/home/minwookje/coding/ex1_w2v/embedding/1542954106final_total_pos.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# word2index table 생성 {token:index}
# 모든 train val test에 사용되는 token별 index table
# train 문장 token 갯수
word_index = len(set_words)
word_tmp_dict = dict()
for i, word in enumerate(set_words):
    word_tmp_dict[str(word).replace(" ", "")] = i
word_tmp_dict['0'] = word_index 

#dict file 저장
with open('token2index.json','w') as dictionary_file:
    json.dump(word_tmp_dict,dictionary_file)

## 문장별 토큰화시킨 녀석에 index를 집어 넣어준다. 이때 pad도 동시에 해준다. 
word_vec = []
word_vec_test = []
word_vec_val = []

for sent in train_tmp:
    sub = []
    for word in sent:
        if(len(sub)==max_count):
            break    
         #print(word)
         #print(type(str(word)))
         #break
# word는 tuple 타입, embeddings_index는 str타입, tuple 타입을 str()화시키면 
# 중간에 space가 생성되어 match가 되지 않았다. 이를 해결해주었다. 
        if(str(word).replace(" ", "") in word_tmp_dict):
            
            sub.append(word_tmp_dict[str(word).replace(" ", "")])
        else:
            print("sentence index화 실패")
    count = max_count - len(sub)
    # padding
    sub.extend([word_index]*count)
    word_vec.append(sub)
## 테스트용 복사본 
for sent in test_tmp:
    sub = []
    for word in sent:
        if(len(sub)==max_count):
            break
         #print(word)
         #print(type(str(word)))
         #break
# word는 tuple 타입, embeddings_index는 str타입, tuple 타입을 str()화시키면 
# 중간에 space가 생성되어 match가 되지 않았다. 이를 해결해주었다. 
        if(str(word).replace(" ", "") in word_tmp_dict):
            
            sub.append(word_tmp_dict[str(word).replace(" ", "")])
        else:
            print("sentence index화 실패")
    count = max_count - len(sub)
    # padding
    sub.extend([word_index]*count)
    word_vec_test.append(sub)

## 검증용 복사본 
for sent in val_tmp:
    sub = []
    for word in sent:
        if(len(sub)==max_count):
            break
         #print(word)
         #print(type(str(word)))
         #break
# word는 tuple 타입, embeddings_index는 str타입, tuple 타입을 str()화시키면 
# 중간에 space가 생성되어 match가 되지 않았다. 이를 해결해주었다. 
        if(str(word).replace(" ", "") in word_tmp_dict):
            
            sub.append(word_tmp_dict[str(word).replace(" ", "")])
        else:
            print("sentence index화 실패")
    count = max_count - len(sub)
    # padding
    sub.extend([word_index]*count)
    word_vec_val.append(sub)

print("word_vec shape:" + str(len(word_vec))+","+ str(len(word_vec[0])))
print("word_vec_val shape:" + str(len(word_vec_val))+","+ str(len(word_vec_val[0])))
print("word_vec_test shape:" + str(len(word_vec_test))+","+ str(len(word_vec_test[0])))




# 4번쨰 matrix embedding_matrix {index: vector}
# vocab_size = min(MAX_NB_WORDS, word_index)
vocab_size = word_index
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size+1, EMB_DIM))
match_count = 0
unmatch_count = 0

for word, i in word_tmp_dict.items():
    if word != '0':
        if (word in embeddings_index):
            match_count += 1
            # embedding_matrix[i] = np.zeros(300)
            embedding_matrix[i] = embeddings_index[word]
            # embedding_matrix[i] = np.random.uniform(-0.25,0.25,300)
        else:
            unmatch_count += 1
            # embedding_matrix[i] = np.zeros(300)
            # embedding_matrix[i] = np.random.uniform(-0.25,0.25,300) ## used for OOV words
            embedding_matrix[i] = np.random.uniform(-1.0,1.0,300).astype('float32')
print("match:" + str(match_count))
print("unmatch:" + str(unmatch_count))

embedding_matrix.tofile('index2vec.dat')
#     여기부터
# 1.앞서 단어당 벡터 테이블(v) // embeddings_index, {w2v_word: vector}
# train_word(str(word).replace(" ", "")) == embedding
# 2.train 단어별 index (v) // word_tmp_dict {train_word(str(word).replace(" ", "")):index}
# 3.sentence padding, sentence to index
# 4.index당 vector table (v) //embedding_matrix {index: vector} 이녀석을 embedding weight에 넣어주어야 한다. 

# 문장 = [index들 나열 ] 
# 즉 embedding_matrix로 index를 seq에 넣어준묹장들을 train에 넣어줘야한다. 
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# np.array화
print("word_vec shape:" + str(len(word_vec))+","+ str(len(word_vec[0])))
print("word_vec_val shape:" + str(len(word_vec_val))+","+ str(len(word_vec_val[0])))
print("word_vec_test shape:" + str(len(word_vec_test))+","+ str(len(word_vec_test[0])))
# np.savetxt('word_vec1.txt', word_vec[:500], delimiter=" ", fmt="%s") 
# np.savetxt('word_vec_val1.txt', word_vec_val[:500], delimiter=" ", fmt="%s") 
# np.savetxt('word_vec_test1.txt', word_vec_test[:500], delimiter=" ", fmt="%s") 

word_vec = np.array(word_vec)
word_vec_val = np.array(word_vec_val)
word_vec_test = np.array(word_vec_test)
embedding_matrix = np.matrix(embedding_matrix)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print("word_vec shape:" + str(word_vec.shape))
print("word_vec_val shape:" + str(word_vec_val.shape))
print("word_vec_test shape:" + str(word_vec_test.shape))
print("y_train shape:" + str(y_train.shape))
print("y_val shape:" + str(y_val.shape))
print("y_test shape:" + str(y_test.shape))
print("embedding_matrix shape:" + str(embedding_matrix.shape))

# 확인용
np.savetxt('word_vec.txt', word_vec[:500], delimiter=" ", fmt="%s") 
np.savetxt('word_vec_val.txt', word_vec_val[:500], delimiter=" ", fmt="%s") 
np.savetxt('word_vec_test.txt', word_vec_test[:500], delimiter=" ", fmt="%s") 
np.savetxt('embedding_matrix.txt', embedding_matrix[:600], delimiter=" ", fmt="%s") 
# TODO
# 1.padding 23976이 너무 많이 들어간다. 이거 max_count를 정해주어야 한다. 100까지로 줄이자.
# 2.그 다음에는 matrix 사이즈를 맞춰주어야 한다. 


# def fit_multilabel(model, X_train, X_val, y_train, y_val):
#     y_val = np.array(y_val)
#     y_train = np.array(y_train)

#     predictions = np.zeros(y_val.shape)

#     for i in range(y_val.shape[1]):
#         model.fit(X_train, y_train[:, i])
#         y_p = model.predict(X_val)
#         predictions[:, i] = y_p

#     return predictions
# from sklearn.metrics import jaccard_similarity_score

# def jaccard_distance_loss(y_true, y_pred, smooth=100):
    # """
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #         = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    # The jaccard distance loss is usefull for unbalanced datasets. This has been
    # shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    # gradient.
    
    # Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    # @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    # @author: wassname
    # """
    # print("loss = %s, %s"%(y_true,y_pred))
    # y_pred = K.cast_to_floatx(y_pred)
    # y_true = K.cast_to_floatx(y_true)
    # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    # sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    # jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # print("loss = %s, %s, %s"%(intersection, sum, jac))   
    # np.zeros_like(y_true, dtype = object)
    # np.zeros_like(y_pred, dtype = object)

    # im1 = np.asarray(y_true).astype(np.bool)
    # im2 = np.asarray(y_pred).astype(np.bool)
    # intersection = np.logical_and(im1, im2)
    # union = np.logical_or(im1, im2)
    # # intersection = K.sum(np.absolute(y_true * y_pred), axis=-1)
    # # sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    # # jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # jac = (intersection.sum()+smooth) / (float(union.sum()+smooth))
    # print("loss = %s, %s, %s"%(intersection, sum, jac))
    # print("intersection")
    # print(intersection)
    # # print(K.eval(intersection))
    # print("sum")
    # print(sum)
    # # print(K.eval(sum))
    # print("jac")
    # print(jac)
    # # print(K.eval(jac))
    # print("K.eval(1-jac)*smooth")
    # print((1 - jac) * smooth)
    # print(type((1 - jac) * smooth))
    # print(K.eval(1-jac)*smooth)
    # returnimport tensorflow as tf (1 - jac) * smooth

# def jaccard_distance(y_true, y_pred, smooth=100):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return (1 - jac) * smooth
#####################################2. 모델 구성하기
# input_length : 단어의 수 즉 문장의 길이를 나타냅니다
# 임베딩 레이어의 출력 크기는 샘플 수 * output_dim * input_lenth가 됩니다

INPUT_DIM = 300 #wordvec사이즈
max_count = max_count #문장의 max길이
lstm_shape = 250
rate_drop_lstm = 0.3
rate_drop_dense = 0.3

# k_vec = K.variable(word_vec)
# k_vec_val = K.variable(word_vec_val)
# k_vec_test = K.variable(word_vec_test)
# k_y_train = K.variable(y_train)
# k_y_val = K.variable(y_val)
# k_y_test = K.variable(y_test)
# k_embedding_matrix = K.variable(embedding_matrix)
k_vec = word_vec
k_vec_val = word_vec_val
k_vec_test = word_vec_test
k_y_train = y_train
k_y_val = y_val
k_y_test = y_test
k_embedding_matrix = embedding_matrix

# nan delete
index = np.argwhere(np.isnan(k_y_train))[:,0]
index2 = np.argwhere(np.isnan(k_y_val))[:,0]
index3 = np.argwhere(np.isnan(k_y_test))[:,0]

k_y_train = np.delete(k_y_train,index,0)
k_vec = np.delete(k_vec,index,0)
k_y_val = np.delete(k_y_val,index2,0)
k_vec_val = np.delete(k_vec_val,index2,0)
k_vec_test = np.delete(k_vec_test,index3,0)
k_y_test = np.delete(k_y_test,index3,0)
# is nan check
print(np.any(np.isnan(k_vec)))
print(np.any(np.isnan(k_vec_val)))
print(np.any(np.isnan(k_vec_test)))
print(np.any(np.isnan(k_y_train)))
print(np.any(np.isnan(k_y_val)))
print(np.any(np.isnan(k_y_test)))
print(np.any(np.isnan(k_embedding_matrix)))



def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """

    epsilon = tf.convert_to_tensor(1e-7, dtype='float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

# matrix accu
def jaccard_distance_acc(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return jac

# def scaled_binary_cross_entropy(y_true,y_pred):
#     epsilon = tf.convert_to_tensor(1e-7, dtype='float32')
#     r =K.binary_crossentropy(y_true,y_pred)
    
#     return r/(r.max()+epsilon)

def scaled_binary_cross_entropy(y_true,y_pred):
    epsilon = tf.convert_to_tensor(1e-7, dtype='float32')
    loss =binary_crossentropy(y_true,y_pred)
    max_t = K.max(loss)
    return loss/(max_t+epsilon)


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))



accu = jaccard_distance_acc    


# 32하니까 죽어버린다.
batch_size = 32
l2_reg = 0.0001
activity_l2 = 0.0001
# Define the model
# inp = InputLayer(shape=(max_count,), dtype='float32',sparse=True)
# input layer는 
# inp = InputLayer(input_shape=(max_count,),sparse=True)

# inp = Input(shape=(max_count,))
inp = Input(shape=(max_count,), dtype='float32')
emb = Embedding(input_dim=vocab_size+1, output_dim=EMB_DIM,
            trainable=False, weights=[embedding_matrix], input_length=max_count)(inp)
# max_features = vocab_size, maxlen=text_max_words, embed_size=EMB_DIM
# emb = Embedding(input_dim=max_features, input_length = maxlen, output_dim=embed_size)(inp)
# embedding dropout = 0.1
x = SpatialDropout1D(0.1)(emb)
# x = Bidirectional(LSTM(lstm_shape, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, W_regularizer=l2(l2_reg)))(x)
# weight, bias, hidden state 수정해보기 , 현재는 kernel만 regularization 시켰다. 
x = Bidirectional(LSTM(lstm_shape, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,kernel_regularizer=l2(l2_reg)))(x)

x, attention = Attention()(x,rate_drop_dense,l2_reg)
# BatchNOrmalization 추가
# x = BatchNormalization()(x)

# 배치 노멀라이제이션 했을때 결과
# Train on 6474 samples, validate on 867 samples
# Epoch 1/50
#   16/6474 [..............................] - ETA: 20:46 - loss: 1.0745 - jaccard  32/6474 [..............................] - ETA: 11:52 - loss: 1.0172 - jaccard  48/6474 [..............................] - ETA: 8:53 - loss: 0.9787 - jaccard_  64/6474 [..............................] - ETA: 7:24 - loss: 0.9671 - jaccard_  80/6474 [..............................] - ETA: 6:30 - loss: 0.9672 - jaccard_  96/6474 [..............................] - ETA: 5:55 - loss: 0.9630 - jaccard_ 112/6474 [..............................] - ETA: 5:29 - loss: 0.9638 - jaccard_ 128/6474 [..............................] - ETA: 5:10 - loss: 0.9557 - jaccard_ 144/6474 [..............................] - ETA: 4:54 - loss: 0.9461 - jaccard_ 160/6474 [..............................] - ETA: 4:42 - loss: 0.9401 - jaccard_ 176/6474 [..............................] - ETA: 4:31 - loss: 0.9331 - jaccard_ 192/6474 [..............................] - ETA: 4:23 - loss: 0.9287 - jaccard_ 208/6474 [..............................] - ETA: 4:15 - loss: 0.9296 - jaccard_ 224/6474 [>.............................] - ETA: 4:09 - loss: 0.9260 - jaccard_ 240/6474 [>.............................] - ETA: 4:03 - loss: 0.9206 - jaccard_ 256/6474 [>.............................] - ETA: 3:59 - loss: 0.9146 - jaccard_ 272/6474 [>.............................] - ETA: 3:54 - loss: 0.9106 - jaccard_ 288/6474 [>.............................] - ETA: 3:50 - loss: 0.9075 - jaccard_ 304/6474 [>.............................] - ETA: 3:47 - loss: 0.9047 - jaccard_ 320/6474 [>.............................] - ETA: 3:44 - loss: 0.9023 - jaccard_ 336/6474 [>.............................] - ETA: 3:41 - loss: 0.8991 - jaccard_ 352/6474 [>.............................] - ETA: 3:38 - loss: 0.8952 - jaccard_ 368/6474 [>.............................] - ETA: 3:36 - loss: 0.8920 - jaccard_ 384/6474 [>.............................] - ETA: 3:34 - loss: 0.8883 - jaccard_ 400/6474 [>.............................] - ETA: 3:32 - loss: 0.8852 - jaccard_ 416/6474 [>.............................] - ETA: 3:29 - loss: 0.8824 - jaccard_ 432/6474 [=>............................] - ETA: 3:28 - loss: 0.8802 - jaccard_ 448/6474 [=>............................] - ETA: 3:27 - loss: 0.8775 - jaccard_ 464/6474 [=>............................] - ETA: 3:25 - loss: 0.8795 - jaccard_ 480/6474 [=>............................] - ETA: 3:23 - loss: 0.8768 - jaccard_ 496/6474 [=>............................] - ETA: 3:22 - loss: 0.8741 - jaccard_ 512/6474 [=>............................] - ETA: 3:20 - loss: 0.8715 - jaccard_ 528/6474 [=>............................] - ETA: 3:18 - loss: 0.8688 - jaccard_ 544/6474 [=>............................] - ETA: 3:17 - loss: 0.8665 - jaccard_ 560/6474 [=>............................] - ETA: 3:15 - loss: 0.8641 - jaccard_ 576/6474 [=>............................] - ETA: 3:14 - loss: 0.8612 - jaccard_ 592/6474 [=>............................] - ETA: 3:13 - loss: 0.8591 - jaccard_ 608/6474 [=>............................] - ETA: 3:11 - loss: 0.8568 - jaccard_ 624/6474 [=>............................] - ETA: 3:10 - loss: 0.8547 - jaccard_ 640/6474 [=>............................] - ETA: 3:09 - loss: 0.8524 - jaccard_ 656/6474 [==>...........................] - ETA: 3:08 - loss: 0.8501 - jaccard_ 672/6474 [==>...........................] - ETA: 3:07 - loss: 0.8476 - jaccard_ 688/6474 [==>...........................] - ETA: 3:05 - loss: 0.8453 - jaccard_ 704/6474 [==>...........................] - ETA: 3:04 - loss: 0.8429 - jaccard_ 720/6474 [==>...........................] - ETA: 3:03 - loss: 0.84 912/6474 [===>..........................] - ETA: 2:53 - loss: 0.8130 - jaccard_distance_acc: 0.1665 - fbeta: 0.2083fbeta: 0.2358
# 6474/6474 [==============================] - 206s 32ms/step - loss: 0.5829 - jaccard_distance_acc: 0.1617 - fbeta: 0.0630 - val_loss: 0.5062 - val_jaccard_distance_acc: 0.1619 - val_fbeta: 0.0180
# Epoch 2/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4948 - jaccard_distance_acc: 0.1611 - fbeta: 0.0348 - val_loss: 0.4914 - val_jaccard_distance_acc: 0.1796 - val_fbeta: 0.0161
# Epoch 3/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4834 - jaccard_distance_acc: 0.1609 - fbeta: 0.0316 - val_loss: 0.4861 - val_jaccard_distance_acc: 0.1738 - val_fbeta: 0.0166
# Epoch 4/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4809 - jaccard_distance_acc: 0.1612 - fbeta: 0.0326 - val_loss: 0.4925 - val_jaccard_distance_acc: 0.1617 - val_fbeta: 0.0176
# Epoch 5/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4811 - jaccard_distance_acc: 0.1614 - fbeta: 0.0344 - val_loss: 0.4842 - val_jaccard_distance_acc: 0.1642 - val_fbeta: 0.0161
# Epoch 6/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4808 - jaccard_distance_acc: 0.1606 - fbeta: 0.0327 - val_loss: 0.4809 - val_jaccard_distance_acc: 0.1715 - val_fbeta: 0.0172
# Epoch 7/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4837 - jaccard_distance_acc: 0.1613 - fbeta: 0.0373 - val_loss: 3.1112 - val_jaccard_distance_acc: 0.2469 - val_fbeta: 0.5642
# Epoch 8/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.5343 - jaccard_distance_acc: 0.1631 - fbeta: 0.0678 - val_loss: 0.5607 - val_jaccard_distance_acc: 0.1851 - val_fbeta: 0.2194
# Epoch 9/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4915 - jaccard_distance_acc: 0.1605 - fbeta: 0.0310 - val_loss: 0.4925 - val_jaccard_distance_acc: 0.1687 - val_fbeta: 0.0161
# Epoch 10/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4810 - jaccard_distance_acc: 0.1611 - fbeta: 0.0325 - val_loss: 0.4936 - val_jaccard_distance_acc: 0.1533 - val_fbeta: 0.0161
# Epoch 11/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4798 - jaccard_distance_acc: 0.1609 - fbeta: 0.0323 - val_loss: 0.4872 - val_jaccard_distance_acc: 0.1572 - val_fbeta: 0.0161
# Epoch 12/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4754 - jaccard_distance_acc: 0.1606 - fbeta: 0.0309 - val_loss: 0.4837 - val_jaccard_distance_acc: 0.1626 - val_fbeta: 0.0161
# Epoch 13/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4821 - jaccard_distance_acc: 0.1606 - fbeta: 0.0315 - val_loss: 0.4865 - val_jaccard_distance_acc: 0.1684 - val_fbeta: 0.0161
# Epoch 14/50
# 6474/6474 [==============================] - 194s 30ms/step - loss: 0.4783 - jaccard_distance_acc: 0.1612 - fbeta: 0.0325 - val_loss: 0.4824 - val_jaccard_distance_acc: 0.1702 - val_fbeta: 0.0161
# Epoch 15/50
# 6474/6474 [==============================] - 186s 29ms/step - loss: 0.4774 - jaccard_distance_acc: 0.1615 - fbeta: 0.0329 - val_loss: 0.4809 - val_jaccard_distance_acc: 0.1674 - val_fbeta: 0.0161
# Epoch 16/50
# 6474/6474 [==============================] - 186s 29ms/step - loss: 0.4771 - jaccard_distance_acc: 0.1609 - fbeta: 0.0309 - val_loss: 0.4808 - val_jaccard_distance_acc: 0.1681 - val_fbeta: 0.0161
# Epoch 17/50
# 6474/6474 [==============================] - 193s 30ms/step - loss: 0.4889 - jaccard_distance_acc: 0.1615 - fbeta: 0.0434 - val_loss: 0.4949 - val_jaccard_distance_acc: 0.1591 - val_fbeta: 0.0161
# Epoch 18/50
# 6474/6474 [==============================] - 186s 29ms/step - loss: 0.5120 - jaccard_distance_acc: 0.1625 - fbeta: 0.0496 - val_loss: 0.5535 - val_jaccard_distance_acc: 0.1604 - val_fbeta: 0.0180
# Epoch 19/50
# 6474/6474 [==============================] - 187s 29ms/step - loss: 0.4868 - jaccard_distance_acc: 0.1611 - fbeta: 0.0316 - val_loss: 0.4873 - val_jaccard_distance_acc: 0.1664 - val_fbeta: 0.0161
# Epoch 20/50
# 6474/6474 [==============================] - 190s 29ms/step - loss: 0.4776 - jaccard_distance_acc: 0.1608 - fbeta: 0.0317 - val_loss: 0.4806 - val_jaccard_distance_acc: 0.1640 - val_fbeta: 0.0161
# Epoch 21/50
# 6474/6474 [==============================] - 201s 31ms/step - loss: 0.4749 - jaccard_distance_acc: 0.1610 - fbeta: 0.0307 - val_loss: 0.4834 - val_jaccard_distance_acc: 0.1651 - val_fbeta: 0.0161
# Epoch 22/50
# 6474/6474 [==============================] - 190s 29ms/step - loss: 0.4760 - jaccard_distance_acc: 0.1608 - fbeta: 0.0307 - val_loss: 0.4805 - val_jaccard_distance_acc: 0.1689 - val_fbeta: 0.0161
# Epoch 23/50
#  144/6474 [..............................] - ETA: 2:58 - loss: 0.4965 - jaccard_distance_acc: 0.1596 - fbeta: 0.0278^Z






# Dense(11, activation="sigmoid",activity_regularizer=activity_l2(0.0001))
x = Dense(11, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

# # 1
# model.compile(loss='binary_crossentropy',
#             optimizer='adam',
#             metrics=[accu])

# Epoch 1/10
#   32/6474 [..............................] - ETA: 9:22 - loss: 0.7045 - jaccard_  64/6474 [..............................] - ETA: 6:18 - loss: 0.6864 - jaccard_  96/6474 [..............................] - ETA: 5:04 - loss: 0.6675 - jaccard_ 128/6474 [..............................] - ETA: 4:24 - loss: 0.6393 - jaccard_ 160/6474 [..............................] - ETA: 3:59 - loss: 0.6099 - jaccard_ 192/6474 [..............................] - ETA: 3:44 - loss: 0.5952 - jaccard_ 224/6474 [>.............................] - ETA: 3:31 - loss: 0.5786 - jaccard_ 256/6474 [>.............................] - ETA: 3:22 - loss: 0.5659 - jaccard_ 288/6474 [>.............................] - ETA: 3:15 - loss: 0.5540 - jaccard_ 320/6474 [>.............................] - ETA: 3:09 - loss: 0.5526 - jaccard_ 352/6474 [>.............................] - ETA: 3:04 - loss: 0.5446 - jaccard_ 384/6474 [>.............................] - ETA: 3:00 - loss: 0.5385 - jaccard_ 416/6474 [>.............................] - ETA: 2:56 - loss: 0.5341 - jaccard_ 448/6474 [=>............................] - ETA: 2:53 - loss: 0.5309 - jaccard_ 480/6474 [=>............................] - ETA: 2:50 - loss: 0.5289 - jaccard_ 512/6474 [=>............................] - ETA: 2:47 - loss: 0.5280 - jaccard_ 544/6474 [=>............................] - ETA: 2:45 - loss: 0.5263 - jaccard_ 576/6474 [=>............................] - ETA: 2:43 - loss: 0.5224 - jaccard_ 608/6474 [=>............................] - ETA: 2:41 - loss: 0.5201 - jaccard_ 640/6474 [=>............................] - ETA: 2:39 - loss: 0.5217 - jaccard_ 672/6474 [==>...........................] - ETA: 2:38 - loss: 0.5178 - jaccard_ 704/6474 [==>...........................] - ETA: 2:37 - loss: 0.5154 - jaccard_ 736/6474 [==>...........................] - ETA: 2:35 - loss: 0.5115 - jaccard_ 768/6474 [==>...........................] - ETA: 2:34 - loss: 0.5113 - jaccard_ 800/6474 [==>...........................] - ETA: 2:32 - loss: 0.5100 - jaccard_ 832/6474 [==>...........................] - ETA: 2:31 - loss: 0.5091 - jaccard_ 864/6474 [===>..........................] - ETA: 2:29 - loss: 0.5059 - jaccard_ 896/6474 [===>..........................] - ETA: 2:28 - loss: 0.5033 - jaccard_ 928/6474 [===>..........................] - ETA: 2:27 - loss: 0.5028 - jaccard_ 2496/6474 [==========>...................] - ETA: 1:39 - loss: 0.4840 - jaccard_distance_acc: 2592/6474 [===========>..................] - ETA: 1:36 - loss: 0.2624/6474 [===========>..................] - ETA: 1:36 - loss: 0.4848 - jaccard_2656/6474 [===========>..................] - ETA: 1:35 - loss: 0.4852 - jaccard_2688/6474 [===========>..................] - ETA: 1:34 - loss: 0.4850 - jaccard_2720/6474 [===========>..................] - ETA: 1:33 - loss: 0.4851 - jaccard_2752/6474 [===========>..................] - ETA: 1:32 - loss: 0.4851 - jaccard_2784/6474 [===========>..................] - ETA: 1:31 - loss: 0.4850 - jaccard_2816/6474 [============>.................] - ETA: 1:31 - loss: 0.4847 - jaccard_2848/6474 [============>.................] - ETA: 1:30 - loss: 0.4848 - jaccard_2880/6474 [============>.................] - ETA: 1:29 - loss: 0.4846 - jaccard_2912/6474 [============>.................] - ETA: 1:28 - loss: 0.4849 - jaccard_2944/6474 [============>.................] - ETA: 1:27 - loss: 0.4845 - jaccard_2976/6474 [=====6474/6474 [==============================] - 165s 26ms/step - loss: 0.4775 - jaccard_distance_acc: 0.1631 - val_loss: 0.4660 - val_jaccard_distance_acc: 0.1825.1623
# Epoch 2/10
# 1408/6474 [=====>........................] - ETA: 2:01 - loss: 0.4571 - jaccard_1440/6474 [=====>........................] - ETA: 2:01 - loss: 0.4582 - jaccard_1472/6474 [=====>........................] - ETA: 2:00 - loss: 0.4585 - jaccard_1504/6474 [=====>........................] - ETA: 1:59 - loss: 0.4579 - jaccard_1536/6474 [======>.......................] - ETA: 1:59 - loss: 0.4578 - jaccard_1568/6474 [======>.......................] - ETA: 1:58 - loss: 0.4573 - jaccard_1600/6474 [======>.......................] - ETA: 1:57 - loss: 0.4581 - jaccard_1632/6474 [======>.......................] - ETA: 1:56 - loss: 0.4579 - jaccard_1664/6474 [======>.......................] - ETA: 1:56 - loss: 0.4582 - jaccard_1696/6474 [======>.......................] - ETA: 1:55 - loss: 0.4581 - jaccard_1728/6474 [=======>......................] - ETA: 1:54 - loss: 0.4584 - jaccard_1760/6474 [=======>......................] - ETA: 1:53 - loss: 0.4583 - jaccard_1792/6474 [=======>......................] - ETA: 1:53 - loss: 0.4582 - jaccard_1824/6474 [=======>......................] - ETA: 1:52 - loss: 0.4584 - jaccard_1856/6474 [=======>......................] - ETA: 1:51 - loss: 0.4581 - jaccard_1888/6474 [=======>......................] - ETA: 1:50 - loss: 0.4575 - jaccard_1920/6474 [=======>......................] - ETA: 1:50 - loss: 0.4587 - jaccard_1952/6474 [========>.....................] - ETA: 1:49 - loss: 0.4585 - jaccard_1984/6474 [========>.....................] - ETA: 1:48 - loss: 0.4582 - jaccard_2016/6474 [========>.....................] - ETA: 1:47 - loss: 0.4582 - jaccard_2048/6474 [========>.....................] - ETA: 1:46 - loss: 0.4579 - jaccard_2080/6474 [========>.....................] - ETA: 1:46 - loss: 0.4578 - jaccard_2112/6474 [========>.....................] - ETA: 1:45 - loss: 0.4576 - jaccard_2144/6474 [========>.....................] - ETA: 1:44 - loss: 0.4573 - jaccard_2176/6474 [=========>....................] - ETA: 1:43 - loss: 0.4570 - jaccard_2208/6474 [=========>....................] - ETA: 1:43 - loss: 0.4568 - jaccard_2240/6474 [=========>.........4608/6474 [====================>.........] - ETA: 45s - loss: 0.4567 - jaccard_distance_acc: 0.6474/6474 [==============================] - 165s 26ms/step - loss: 0.4560 - jaccard_distance_acc: 0.1827 - val_loss: 0.4383 - val_jaccard_distance_acc: 0.2132
# Epoch 3/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.4378 - jaccard_distance_acc: 0.2064 - val_loss: 0.4245 - val_jaccard_distance_acc: 0.2289
# Epoch 4/10
# 6474/6474 [==============================] - 163s 25ms/step - loss: 0.4251 - jaccard_distance_acc: 0.2249 - val_loss: 0.4217 - val_jaccard_distance_acc: 0.2414
# Epoch 5/10
# 6474/6474 [==============================] - 2968s 458ms/step - loss: 0.4117 - jaccard_distance_acc: 0.2425 - val_loss: 0.4122 - val_jaccard_distance_acc: 0.2415
# Epoch 6/10
# 6474/6474 [==============================] - 168s 26ms/step - loss: 0.4013 - jaccard_distance_acc: 0.2553 - val_loss: 0.4052 - val_jaccard_distance_acc: 0.2642
# Epoch 7/10
# 6474/6474 [==============================] - 169s 26ms/step - loss: 0.3919 - jaccard_distance_acc: 0.2667 - val_loss: 0.4075 - val_jaccard_distance_acc: 0.2673
# Epoch 8/10
# 6474/6474 [==============================] - 163s 25ms/step - loss: 0.3826 - jaccard_distance_acc: 0.2808 - val_loss: 0.4040 - val_jaccard_distance_acc: 0.2709
# Epoch 9/10
# 6474/6474 [==============================] - 163s 25ms/step - loss: 0.3721 - jaccard_distance_acc: 0.2928 - val_loss: 0.4054 - val_jaccard_distance_acc: 0.2911
# Epoch 10/10
# 6474/6474 [==============================] - 163s 25ms/step - loss: 0.3616 - jaccard_distance_acc: 0.3084 - val_loss: 0.4056 - val_jaccard_distance_acc: 0.2884
# 3151/3151 [==============================] - 26s 8ms/step


# 2

# model.compile(loss=jaccard_distance_loss,
#             optimizer='adam',
#             metrics=[accu])
# Train on 6474 samples, validate on 867 samples
# Epoch 1/10
# 6474/6474 [==============================] - 164s 25ms/step - loss: 0.6940 - jaccard_distance_acc: 0.3060 - val_loss: 0.6702 - val_jaccard_distance_acc: 0.3298
# Epoch 2/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.6902 - jaccard_distance_acc: 0.3098 - val_loss: 0.6702 - val_jaccard_distance_acc: 0.3298
# Epoch 3/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.6902 - jaccard_distance_acc: 0.3098 - val_loss: 0.6702 - val_jaccard_distance_acc: 0.3298
# Epoch 4/10
# 2144/6474 [========>.....................] - ETA: 1:43 - loss: 0.6901 - jaccard_distance_acc: 0.3099^Z
# [1]+  정지됨               python3 181120_preprocessing.py


# 3
# model.compile(loss='binary_crossentropy',
#             optimizer='adam',
#             metrics=['accracy'])
# Train on 6474 samples, validate on 867 samples
# Epoch 1/10
# 6474/6474 [==============================] - 163s 25ms/step - loss: 0.4774 - acc: 0.7840 - val_loss: 0.4684 - val_acc: 0.7794
# Epoch 2/10
# 6474/6474 [==============================] - 161s 25ms/step - loss: 0.4563 - acc: 0.7935 - val_loss: 0.4335 - val_acc: 0.8041
# Epoch 3/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.4369 - acc: 0.8056 - val_loss: 0.4264 - val_acc: 0.8064
# Epoch 4/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.4242 - acc: 0.8110 - val_loss: 0.4200 - val_acc: 0.8087
# Epoch 5/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.4104 - acc: 0.8174 - val_loss: 0.4119 - val_acc: 0.8171
# Epoch 6/10
# 6474/6474 [==============================] - 162s 25ms/step - loss: 0.4017 - acc: 0.8241 - val_loss: 0.4063 - val_acc: 0.8188
# Epoch 7/10
# 6474/6474 [==============================] - 730s 113ms/step - loss: 0.3942 - acc: 0.8273 - val_loss: 0.4063 - val_acc: 0.8184
# Epoch 8/10
# 6474/6474 [==============================] - 168s 26ms/step - loss: 0.3857 - acc: 0.8318 - val_loss: 0.4036 - val_acc: 0.8199
# Epoch 9/10
# 6474/6474 [==============================] - 164s 25ms/step - loss: 0.3777 - acc: 0.8355 - val_loss: 0.4021 - val_acc: 0.8219
# Epoch 10/10
# 6474/6474 [==============================] - 179s 28ms/step - loss: 0.3676 - acc: 0.8384 - val_loss: 0.4051 - val_acc: 0.8204
# 3151/3151 [==============================] - 27s 9ms/step

# score : 0.406278005236acc : 0.823230730148

# 4
# model.compile(loss='binary_crossentropy',
#             optimizer='rmsprop',
#             metrics=['acc', accu])
# Epoch 1/50
#   16/6474 [..............................] - ETA: 31:06 - loss: 0.6962 - jaccard_distance_acc: 0.1575 - f  32/6474 [..............................] - ETA: 18:30 - loss: 0.6798 - jaccard_distance_acc: 0.1541 - f  48/6474 [..............................] - ETA: 14:12 - loss: 0.6345 - jaccard_distance_acc: 0.1517 - f  64/6474 [..............................] - ETA: 12:08 - loss: 0.6052 - jaccard_distance_acc: 0.1490 - f  80/6474 [..............................] - ETA: 10:58 - loss: 0.5777 - jaccard_distance_acc: 0.1464 - f  96/6474 [..............................] - ETA: 10:07 - loss: 0.5617 - jaccard_distance_acc: 0.1479 - f 112/6474 [..............................] - ETA: 9:29 - loss: 0.5565 - jaccard_distance_acc: 0.1488 - fb 128/6474 [..............................] - ETA: 9:03 - loss: 0.5461 - jaccard_distance_acc: 0.1512 - fb 144/6474 [..............................] - ETA: 8:39 - loss: 0.5464 - jaccard_distance_acc: 0.1553 - fb 160/6474 [..............................] - ETA: 8:21 - loss: 0.5346 - jaccard_distance_acc: 0.1541 - fb 176/6474 [..............................] - ETA: 8:07 - loss: 0.5358 - jaccard_distance_acc: 0.1555 - fb 192/6474 [..............................] - ETA: 7:53 - loss: 0.5330 - jaccard_distance_acc: 0.1548 - fb 208/6474 [..............................] - ETA: 7:41 - loss: 0.5300 - jaccard_distance_acc: 0.1539 - fb 224/6474 [>.............................] - ETA: 7:31 - loss: 0.5278 - jaccard_distance_acc: 0.1524 - fb 240/6474 [>.............................] - ETA: 7:22 - loss: 0.5257 - jaccard_distance_acc: 0.1515 - fb 256/6474 [>.............................] - ETA: 7:16 - loss: 0.5219 - jaccard_distance_acc: 0.1515 - fb 272/6476474/6474 [==============================] - 377s 58ms/step - loss: 0.4761 - jaccard_distance_acc: 0.1626 - fbeta: 0.0427 - val_loss: 0.4718 - val_jaccard_distance_acc: 0.1743 - val_fbeta: 0.0150
# Epoch 2/50
# 6474/6474 [==============================] - 367s 57ms/step - loss: 0.4506 - jaccard_distance_acc: 0.1899 - fbeta: 0.1640 - val_loss: 0.4286 - val_jaccard_distance_acc: 0.2238 - val_fbeta: 0.2165
# Epoch 3/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4303 - jaccard_distance_acc: 0.2180 - fbeta: 0.2692 - val_loss: 0.4274 - val_jaccard_distance_acc: 0.2380 - val_fbeta: 0.2963
# Epoch 4/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4167 - jaccard_distance_acc: 0.2362 - fbeta: 0.3153 - val_loss: 0.4138 - val_jaccard_distance_acc: 0.2498 - val_fbeta: 0.3165
# Epoch 5/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4004 - jaccard_distance_acc: 0.2586 - fbeta: 0.3697 - val_loss: 0.4156 - val_jaccard_distance_acc: 0.2428 - val_fbeta: 0.2902
# Epoch 6/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3892 - jaccard_distance_acc: 0.2726 - fbeta: 0.3920 - val_loss: 0.4030 - val_jaccard_distance_acc: 0.2778 - val_fbeta: 0.3690
# Epoch 7/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3776 - jaccard_distance_acc: 0.2876 - fbeta: 0.4294 - val_loss: 0.4071 - val_jaccard_distance_acc: 0.2727 - val_fbeta: 0.3754
# Epoch 8/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3661 - jaccard_distance_acc: 0.3027 - fbeta: 0.4526 - val_loss: 0.4072 - val_jaccard_distance_acc: 0.2715 - val_fbeta: 0.3536
# Epoch 9/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3536 - jaccard_distance_acc: 0.3184 - fbeta: 0.4749 - val_loss: 0.4045 - val_jaccard_distance_acc: 0.2980 - val_fbeta: 0.4100
# Epoch 10/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3456 - jaccard_distance_acc: 0.3320 - fbeta: 0.4995 - val_loss: 0.4084 - val_jaccard_distance_acc: 0.3023 - val_fbeta: 0.4141
# Epoch 11/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3350 - jaccard_distance_acc: 0.3447 - fbeta: 0.5130 - val_loss: 0.4093 - val_jaccard_distance_acc: 0.2999 - val_fbeta: 0.4063
# Epoch 12/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3227 - jaccard_distance_acc: 0.3613 - fbeta: 0.5413 - val_loss: 0.4135 - val_jaccard_distance_acc: 0.3017 - val_fbeta: 0.4054
# Epoch 13/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.3143 - jaccard_distance_acc: 0.3726 - fbeta: 0.5565 - val_loss: 0.4175 - val_jaccard_distance_acc: 0.3082 - val_fbeta: 0.4300
# Epoch 14/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3070 - jaccard_distance_acc: 0.3844 - fbeta: 0.5686 - val_loss: 0.4289 - val_jaccard_distance_acc: 0.3154 - val_fbeta: 0.4311
# Epoch 15/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2980 - jaccard_distance_acc: 0.3942 - fbeta: 0.5858 - val_loss: 0.4412 - val_jaccard_distance_acc: 0.3166 - val_fbeta: 0.4379
# Epoch 16/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2850 - jaccard_distance_acc: 0.4098 - fbeta: 0.6062 - val_loss: 0.4378 - val_jaccard_distance_acc: 0.3181 - val_fbeta: 0.4349
# Epoch 17/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2803 - jaccard_distance_acc: 0.4209 - fbeta: 0.6202 - val_loss: 0.4379 - val_jaccard_distance_acc: 0.3192 - val_fbeta: 0.4295
# Epoch 18/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2729 - jaccard_distance_acc: 0.4319 - fbeta: 0.6358 - val_loss: 0.4504 - val_jaccard_distance_acc: 0.3201 - val_fbeta: 0.4383
# Epoch 19/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2633 - jaccard_distance_acc: 0.4444 - fbeta: 0.6461 - val_loss: 0.4479 - val_jaccard_distance_acc: 0.3229 - val_fbeta: 0.4409
# Epoch 20/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.2576 - jaccard_distance_acc: 0.4531 - fbeta: 0.6581 - val_loss: 0.4578 - val_jaccard_distance_acc: 0.3308 - val_fbeta: 0.4555
# Epoch 21/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2511 - jaccard_distance_acc: 0.4639 - fbeta: 0.6706 - val_loss: 0.4673 - val_jaccard_distance_acc: 0.3209 - val_fbeta: 0.4305
# Epoch 22/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2436 - jaccard_distance_acc: 0.4743 - fbeta: 0.6839 - val_loss: 0.4847 - val_jaccard_distance_acc: 0.3328 - val_fbeta: 0.4618
# Epoch 23/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2365 - jaccard_distance_acc: 0.4849 - fbeta: 0.6919 - val_loss: 0.4871 - val_jaccard_distance_acc: 0.3315 - val_fbeta: 0.4600
# Epoch 24/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2320 - jaccard_distance_acc: 0.4910 - fbeta: 0.7011 - val_loss: 0.4853 - val_jaccard_distance_acc: 0.3275 - val_fbeta: 0.4493
# Epoch 25/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2269 - jaccard_distance_acc: 0.4989 - fbeta: 0.7053 - val_loss: 0.4953 - val_jaccard_distance_acc: 0.3331 - val_fbeta: 0.4499
# Epoch 26/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2174 - jaccard_distance_acc: 0.5130 - fbeta: 0.7232 - val_loss: 0.4959 - val_jaccard_distance_acc: 0.3457 - val_fbeta: 0.4814
# Epoch 27/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.2148 - jaccard_distance_acc: 0.5194 - fbeta: 0.7295 - val_loss: 0.5119 - val_jaccard_distance_acc: 0.3323 - val_fbeta: 0.4494
# Epoch 28/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2082 - jaccard_distance_acc: 0.5294 - fbeta: 0.7392 - val_loss: 0.5138 - val_jaccard_distance_acc: 0.3390 - val_fbeta: 0.4622
# Epoch 29/50
# 6474/6474 [==============================] - 361s 56ms/step - loss: 0.2031 - jaccard_distance_acc: 0.5345 - fbeta: 0.7428 - val_loss: 0.5134 - val_jaccard_distance_acc: 0.3390 - val_fbeta: 0.4613
# Epoch 30/50
# 6474/6474 [==============================] - 361s 56ms/step - loss: 0.1986 - jaccard_distance_acc: 0.5456 - fbeta: 0.7544 - val_loss: 0.5245 - val_jaccard_distance_acc: 0.3289 - val_fbeta: 0.4362
# Epoch 31/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1950 - jaccard_distance_acc: 0.5508 - fbeta: 0.7570 - val_loss: 0.5260 - val_jaccard_distance_acc: 0.3360 - val_fbeta: 0.4513
# Epoch 32/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1901 - jaccard_distance_acc: 0.5582 - fbeta: 0.7635 - val_loss: 0.5358 - val_jaccard_distance_acc: 0.3401 - val_fbeta: 0.4495
# Epoch 33/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1874 - jaccard_distance_acc: 0.5636 - fbeta: 0.7708 - val_loss: 0.5402 - val_jaccard_distance_acc: 0.3318 - val_fbeta: 0.4381
# Epoch 34/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.1830 - jaccard_distance_acc: 0.5722 - fbeta: 0.7789 - val_loss: 0.5433 - val_jaccard_distance_acc: 0.3342 - val_fbeta: 0.4462
# Epoch 35/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.1787 - jaccard_distance_acc: 0.5776 - fbeta: 0.7838 - val_loss: 0.5424 - val_jaccard_distance_acc: 0.3500 - val_fbeta: 0.4689
# Epoch 36/50
# 6474/6474 [==============================] - 341s 53ms/step - loss: 0.1774 - jaccard_distance_acc: 0.5819 - fbeta: 0.7876 - val_loss: 0.5405 - val_jaccard_distance_acc: 0.3441 - val_fbeta: 0.4530
# Epoch 37/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1704 - jaccard_distance_acc: 0.5920 - fbeta: 0.7924 - val_loss: 0.5666 - val_jaccard_distance_acc: 0.3288 - val_fbeta: 0.4324
# Epoch 38/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1692 - jaccard_distance_acc: 0.5960 - fbeta: 0.7988 - val_loss: 0.5631 - val_jaccard_distance_acc: 0.3399 - val_fbeta: 0.4492
# Epoch 39/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1657 - jaccard_distance_acc: 0.6000 - fbeta: 0.8028 - val_loss: 0.5748 - val_jaccard_distance_acc: 0.3386 - val_fbeta: 0.4434
# Epoch 40/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1635 - jaccard_distance_acc: 0.6059 - fbeta: 0.8055 - val_loss: 0.5664 - val_jaccard_distance_acc: 0.3453 - val_fbeta: 0.4482
# Epoch 41/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1621 - jaccard_distance_acc: 0.6110 - fbeta: 0.8102 - val_loss: 0.5700 - val_jaccard_distance_acc: 0.3384 - val_fbeta: 0.4450
# Epoch 42/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1555 - jaccard_distance_acc: 0.6179 - fbeta: 0.8133 - val_loss: 0.5807 - val_jaccard_distance_acc: 0.3462 - val_fbeta: 0.4560
# Epoch 43/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1540 - jaccard_distance_acc: 0.6219 - fbeta: 0.8187 - val_loss: 0.5937 - val_jaccard_distance_acc: 0.3437 - val_fbeta: 0.4520
# Epoch 44/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1526 - jaccard_distance_acc: 0.6263 - fbeta: 0.8194 - val_loss: 0.5912 - val_jaccard_distance_acc: 0.3377 - val_fbeta: 0.4440
# Epoch 45/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1491 - jaccard_distance_acc: 0.6319 - fbeta: 0.8246 - val_loss: 0.6023 - val_jaccard_distance_acc: 0.3425 - val_fbeta: 0.4602
# Epoch 46/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1453 - jaccard_distance_acc: 0.6380 - fbeta: 0.8269 - val_loss: 0.5966 - val_jaccard_distance_acc: 0.3441 - val_fbeta: 0.4441
# Epoch 47/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1446 - jaccard_distance_acc: 0.6409 - fbeta: 0.8328 - val_loss: 0.5986 - val_jaccard_distance_acc: 0.3503 - val_fbeta: 0.4610
# Epoch 48/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1432 - jaccard_distance_acc: 0.6418 - fbeta: 0.8332 - val_loss: 0.6044 - val_jaccard_distance_acc: 0.3531 - val_fbeta: 0.4591
# Epoch 49/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1396 - jaccard_distance_acc: 0.6516 - fbeta: 0.8384 - val_loss: 0.6242 - val_jaccard_distance_acc: 0.3518 - val_fbeta: 0.4558
# Epoch 50/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1382 - jaccard_distance_acc: 0.6537 - fbeta: 0.8399 - val_loss: 0.6139 - val_jaccard_distance_acc: 0.3450 - val_fbeta: 0.4513
# 3151/3151 [==============================] - 27s 8ms/step




# attention_model = Model(inputs=inp, outputs=attention) # Model to print out the attention data
# model.summary()
# verbose= ? , validation_split은 validation file로 변환시켜주어야 한다.

# 5
# model.compile(loss='binary_crossentropy',
#             optimizer=Adam(clipnorm=1, lr=0.001),
#             metrics=[accu, fbeta])

# Epoch 1/50
#   16/6474 [..............................] - ETA: 31:06 - loss: 0.6962 - jaccard_distance_acc: 0.1575 - f  32/6474 [..............................] - ETA: 18:30 - loss: 0.6798 - jaccard_distance_acc: 0.1541 - f  48/6474 [..............................] - ETA: 14:12 - loss: 0.6345 - jaccard_distance_acc: 0.1517 - f  64/6474 [..............................] - ETA: 12:08 - loss: 0.6052 - jaccard_distance_acc: 0.1490 - f  80/6474 [..............................] - ETA: 10:58 - loss: 0.5777 - jaccard_distance_acc: 0.1464 - f  96/6474 [..............................] - ETA: 10:07 - loss: 0.5617 - jaccard_distance_acc: 0.1479 - f 112/6474 [..............................] - ETA: 9:29 - loss: 0.5565 - jaccard_distance_acc: 0.1488 - fb 128/6474 [..............................] - ETA: 9:03 - loss: 0.5461 - jaccard_distance_acc: 0.1512 - fb 144/6474 [..............................] - ETA: 8:39 - loss: 0.5464 - jaccard_distance_acc: 0.1553 - fb 160/6474 [..............................] - ETA: 8:21 - loss: 0.5346 - jaccard_distance_acc: 0.1541 - fb 176/6474 [..............................] - ETA: 8:07 - loss: 0.5358 - jaccard_distance_acc: 0.1555 - fb 192/6474 [..............................] - ETA: 7:53 - loss: 0.5330 - jaccard_distance_acc: 0.1548 - fb 208/6474 [..............................] - ETA: 7:41 - loss: 0.5300 - jaccard_distance_acc: 0.1539 - fb 224/6474 [>.............................] - ETA: 7:31 - loss: 0.5278 - jaccard_distance_acc: 0.1524 - fb 240/6474 [>.............................] - ETA: 7:22 - loss: 0.5257 - jaccard_distance_acc: 0.1515 - fb 256/6474 [>.............................] - ETA: 7:16 - loss: 0.5219 - jaccard_distance_acc: 0.1515 - fb 272/6476474/6474 [==============================] - 377s 58ms/step - loss: 0.4761 - jaccard_distance_acc: 0.1626 - fbeta: 0.0427 - val_loss: 0.4718 - val_jaccard_distance_acc: 0.1743 - val_fbeta: 0.0150
# Epoch 2/50
# 6474/6474 [==============================] - 367s 57ms/step - loss: 0.4506 - jaccard_distance_acc: 0.1899 - fbeta: 0.1640 - val_loss: 0.4286 - val_jaccard_distance_acc: 0.2238 - val_fbeta: 0.2165
# Epoch 3/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4303 - jaccard_distance_acc: 0.2180 - fbeta: 0.2692 - val_loss: 0.4274 - val_jaccard_distance_acc: 0.2380 - val_fbeta: 0.2963
# Epoch 4/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4167 - jaccard_distance_acc: 0.2362 - fbeta: 0.3153 - val_loss: 0.4138 - val_jaccard_distance_acc: 0.2498 - val_fbeta: 0.3165
# Epoch 5/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.4004 - jaccard_distance_acc: 0.2586 - fbeta: 0.3697 - val_loss: 0.4156 - val_jaccard_distance_acc: 0.2428 - val_fbeta: 0.2902
# Epoch 6/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3892 - jaccard_distance_acc: 0.2726 - fbeta: 0.3920 - val_loss: 0.4030 - val_jaccard_distance_acc: 0.2778 - val_fbeta: 0.3690
# Epoch 7/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3776 - jaccard_distance_acc: 0.2876 - fbeta: 0.4294 - val_loss: 0.4071 - val_jaccard_distance_acc: 0.2727 - val_fbeta: 0.3754
# Epoch 8/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3661 - jaccard_distance_acc: 0.3027 - fbeta: 0.4526 - val_loss: 0.4072 - val_jaccard_distance_acc: 0.2715 - val_fbeta: 0.3536
# Epoch 9/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3536 - jaccard_distance_acc: 0.3184 - fbeta: 0.4749 - val_loss: 0.4045 - val_jaccard_distance_acc: 0.2980 - val_fbeta: 0.4100
# Epoch 10/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3456 - jaccard_distance_acc: 0.3320 - fbeta: 0.4995 - val_loss: 0.4084 - val_jaccard_distance_acc: 0.3023 - val_fbeta: 0.4141
# Epoch 11/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3350 - jaccard_distance_acc: 0.3447 - fbeta: 0.5130 - val_loss: 0.4093 - val_jaccard_distance_acc: 0.2999 - val_fbeta: 0.4063
# Epoch 12/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.3227 - jaccard_distance_acc: 0.3613 - fbeta: 0.5413 - val_loss: 0.4135 - val_jaccard_distance_acc: 0.3017 - val_fbeta: 0.4054
# Epoch 13/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.3143 - jaccard_distance_acc: 0.3726 - fbeta: 0.5565 - val_loss: 0.4175 - val_jaccard_distance_acc: 0.3082 - val_fbeta: 0.4300
# Epoch 14/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.3070 - jaccard_distance_acc: 0.3844 - fbeta: 0.5686 - val_loss: 0.4289 - val_jaccard_distance_acc: 0.3154 - val_fbeta: 0.4311
# Epoch 15/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2980 - jaccard_distance_acc: 0.3942 - fbeta: 0.5858 - val_loss: 0.4412 - val_jaccard_distance_acc: 0.3166 - val_fbeta: 0.4379
# Epoch 16/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2850 - jaccard_distance_acc: 0.4098 - fbeta: 0.6062 - val_loss: 0.4378 - val_jaccard_distance_acc: 0.3181 - val_fbeta: 0.4349
# Epoch 17/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2803 - jaccard_distance_acc: 0.4209 - fbeta: 0.6202 - val_loss: 0.4379 - val_jaccard_distance_acc: 0.3192 - val_fbeta: 0.4295
# Epoch 18/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2729 - jaccard_distance_acc: 0.4319 - fbeta: 0.6358 - val_loss: 0.4504 - val_jaccard_distance_acc: 0.3201 - val_fbeta: 0.4383
# Epoch 19/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2633 - jaccard_distance_acc: 0.4444 - fbeta: 0.6461 - val_loss: 0.4479 - val_jaccard_distance_acc: 0.3229 - val_fbeta: 0.4409
# Epoch 20/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.2576 - jaccard_distance_acc: 0.4531 - fbeta: 0.6581 - val_loss: 0.4578 - val_jaccard_distance_acc: 0.3308 - val_fbeta: 0.4555
# Epoch 21/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2511 - jaccard_distance_acc: 0.4639 - fbeta: 0.6706 - val_loss: 0.4673 - val_jaccard_distance_acc: 0.3209 - val_fbeta: 0.4305
# Epoch 22/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2436 - jaccard_distance_acc: 0.4743 - fbeta: 0.6839 - val_loss: 0.4847 - val_jaccard_distance_acc: 0.3328 - val_fbeta: 0.4618
# Epoch 23/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2365 - jaccard_distance_acc: 0.4849 - fbeta: 0.6919 - val_loss: 0.4871 - val_jaccard_distance_acc: 0.3315 - val_fbeta: 0.4600
# Epoch 24/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2320 - jaccard_distance_acc: 0.4910 - fbeta: 0.7011 - val_loss: 0.4853 - val_jaccard_distance_acc: 0.3275 - val_fbeta: 0.4493
# Epoch 25/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2269 - jaccard_distance_acc: 0.4989 - fbeta: 0.7053 - val_loss: 0.4953 - val_jaccard_distance_acc: 0.3331 - val_fbeta: 0.4499
# Epoch 26/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.2174 - jaccard_distance_acc: 0.5130 - fbeta: 0.7232 - val_loss: 0.4959 - val_jaccard_distance_acc: 0.3457 - val_fbeta: 0.4814
# Epoch 27/50
# 6474/6474 [==============================] - 364s 56ms/step - loss: 0.2148 - jaccard_distance_acc: 0.5194 - fbeta: 0.7295 - val_loss: 0.5119 - val_jaccard_distance_acc: 0.3323 - val_fbeta: 0.4494
# Epoch 28/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.2082 - jaccard_distance_acc: 0.5294 - fbeta: 0.7392 - val_loss: 0.5138 - val_jaccard_distance_acc: 0.3390 - val_fbeta: 0.4622
# Epoch 29/50
# 6474/6474 [==============================] - 361s 56ms/step - loss: 0.2031 - jaccard_distance_acc: 0.5345 - fbeta: 0.7428 - val_loss: 0.5134 - val_jaccard_distance_acc: 0.3390 - val_fbeta: 0.4613
# Epoch 30/50
# 6474/6474 [==============================] - 361s 56ms/step - loss: 0.1986 - jaccard_distance_acc: 0.5456 - fbeta: 0.7544 - val_loss: 0.5245 - val_jaccard_distance_acc: 0.3289 - val_fbeta: 0.4362
# Epoch 31/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1950 - jaccard_distance_acc: 0.5508 - fbeta: 0.7570 - val_loss: 0.5260 - val_jaccard_distance_acc: 0.3360 - val_fbeta: 0.4513
# Epoch 32/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1901 - jaccard_distance_acc: 0.5582 - fbeta: 0.7635 - val_loss: 0.5358 - val_jaccard_distance_acc: 0.3401 - val_fbeta: 0.4495
# Epoch 33/50
# 6474/6474 [==============================] - 362s 56ms/step - loss: 0.1874 - jaccard_distance_acc: 0.5636 - fbeta: 0.7708 - val_loss: 0.5402 - val_jaccard_distance_acc: 0.3318 - val_fbeta: 0.4381
# Epoch 34/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.1830 - jaccard_distance_acc: 0.5722 - fbeta: 0.7789 - val_loss: 0.5433 - val_jaccard_distance_acc: 0.3342 - val_fbeta: 0.4462
# Epoch 35/50
# 6474/6474 [==============================] - 363s 56ms/step - loss: 0.1787 - jaccard_distance_acc: 0.5776 - fbeta: 0.7838 - val_loss: 0.5424 - val_jaccard_distance_acc: 0.3500 - val_fbeta: 0.4689
# Epoch 36/50
# 6474/6474 [==============================] - 341s 53ms/step - loss: 0.1774 - jaccard_distance_acc: 0.5819 - fbeta: 0.7876 - val_loss: 0.5405 - val_jaccard_distance_acc: 0.3441 - val_fbeta: 0.4530
# Epoch 37/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1704 - jaccard_distance_acc: 0.5920 - fbeta: 0.7924 - val_loss: 0.5666 - val_jaccard_distance_acc: 0.3288 - val_fbeta: 0.4324
# Epoch 38/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1692 - jaccard_distance_acc: 0.5960 - fbeta: 0.7988 - val_loss: 0.5631 - val_jaccard_distance_acc: 0.3399 - val_fbeta: 0.4492
# Epoch 39/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1657 - jaccard_distance_acc: 0.6000 - fbeta: 0.8028 - val_loss: 0.5748 - val_jaccard_distance_acc: 0.3386 - val_fbeta: 0.4434
# Epoch 40/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1635 - jaccard_distance_acc: 0.6059 - fbeta: 0.8055 - val_loss: 0.5664 - val_jaccard_distance_acc: 0.3453 - val_fbeta: 0.4482
# Epoch 41/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1621 - jaccard_distance_acc: 0.6110 - fbeta: 0.8102 - val_loss: 0.5700 - val_jaccard_distance_acc: 0.3384 - val_fbeta: 0.4450
# Epoch 42/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1555 - jaccard_distance_acc: 0.6179 - fbeta: 0.8133 - val_loss: 0.5807 - val_jaccard_distance_acc: 0.3462 - val_fbeta: 0.4560
# Epoch 43/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1540 - jaccard_distance_acc: 0.6219 - fbeta: 0.8187 - val_loss: 0.5937 - val_jaccard_distance_acc: 0.3437 - val_fbeta: 0.4520
# Epoch 44/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1526 - jaccard_distance_acc: 0.6263 - fbeta: 0.8194 - val_loss: 0.5912 - val_jaccard_distance_acc: 0.3377 - val_fbeta: 0.4440
# Epoch 45/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1491 - jaccard_distance_acc: 0.6319 - fbeta: 0.8246 - val_loss: 0.6023 - val_jaccard_distance_acc: 0.3425 - val_fbeta: 0.4602
# Epoch 46/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1453 - jaccard_distance_acc: 0.6380 - fbeta: 0.8269 - val_loss: 0.5966 - val_jaccard_distance_acc: 0.3441 - val_fbeta: 0.4441
# Epoch 47/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1446 - jaccard_distance_acc: 0.6409 - fbeta: 0.8328 - val_loss: 0.5986 - val_jaccard_distance_acc: 0.3503 - val_fbeta: 0.4610
# Epoch 48/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1432 - jaccard_distance_acc: 0.6418 - fbeta: 0.8332 - val_loss: 0.6044 - val_jaccard_distance_acc: 0.3531 - val_fbeta: 0.4591
# Epoch 49/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1396 - jaccard_distance_acc: 0.6516 - fbeta: 0.8384 - val_loss: 0.6242 - val_jaccard_distance_acc: 0.3518 - val_fbeta: 0.4558
# Epoch 50/50
# 6474/6474 [==============================] - 185s 29ms/step - loss: 0.1382 - jaccard_distance_acc: 0.6537 - fbeta: 0.8399 - val_loss: 0.6139 - val_jaccard_distance_acc: 0.3450 - val_fbeta: 0.4513
# 3151/3151 [==============================] - 27s 8ms/step

# 6
# model.compile(loss=scaled_binary_cross_entropy,
#             optimizer=Adam(clipnorm=1, lr=0.001),
#             metrics=[accu, fbeta])

# 6' clipnorm=1의 역활이 득이 되는지 아닌지 잘 모르겠다. 평가의 기준이 명확하지 않으니까 
# 내가 맞게 평가 하고 학습하는지 잘 모르겠다. 
# model.compile(loss=scaled_binary_cross_entropy,
#             optimizer=Adam(lr=0.001),
#             metrics=[accu, fbeta])


#7
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=0.001),
            metrics=[accu, fbeta])

attention_model = Model(inputs=inp, outputs=attention) # Model to print out the attention data
model.summary()


# STAMP = './First_AttentionNLP_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
# print(STAMP)
# early_stopping =EarlyStopping(monitor='val_loss', patience=5)
# bst_model_path = STAMP + '.h5'
# model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
# # model.fit(X_t, y, validation_data=(x_val,y_val), epochs=3, verbose=1, batch_size=512)
# epoch = 50




filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    
callbacks_list = [checkpoint]     
model.fit(k_vec, k_y_train, validation_data=(k_vec_val,k_y_val), epochs=30, verbose=1, batch_size=batch_size,callbacks=callbacks_list)


# model.fit(k_vec, k_y_train, validation_data=(k_vec_val,k_y_val), epochs=3, verbose=1, steps_per_epoch=int(6619/batch_size)+1, validation_steps = int(883/batch_size)+1)

# 5. 모델 평가하기
test_score = model.evaluate(k_vec_test, k_y_test, batch_size=batch_size)
print('')
print(str(test_score))



# 6. 모델 저장하기
from datetime import datetime
# now = datetime.now()
model.save('Attention.h5')


def test_return():
    return (k_vec_test,k_y_test)

# #  7. 모델 아키텍처 보기
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# # %matplotlib inline
# # str(now.day)+"/"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# #######################################
# ## make the submission
# ########################################
# print('Start making the submission before fine-tuning')


# y_test = model.predict(word_vec_test,y_test_size=32, verbose=1)

# sample_submission = pd.read_csv("../input/sample_submission.csv")
# sample_submission[list_classes] = y_test

# sample_submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)




# # 테스트용 코드 
# def get_word_importances(text):
#     lt = tokenizer.texts_to_sequences([text])
#     x = pad_sequences(lt, maxlen=maxlen)
#     p = model.predict(x)
#     att = attention_model.predict(x)
#     return p, [(reverse_token_map.get(word), importance) for word, importance in zip(x[0], att[0]) if word in reverse_token_map]












# emb = Embedding(input_dim=vocab_size, output_dim=EMB_DIM,
#                 trainable=False, weights=[embedding_matrix], input_length=text_max_words)

# # 임베딩 dropout

# # GaussianNoise(mean=0.0, stddev=0.2)
# keras.layers.GaussianNoise(stddev=0.2)
# keras.layers.GaussianDropout(rate)

# # LSTM(310, 250, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
# # Dropout(p=0.3)
# # n_timesteps = text_max_words
# # return_sequences = True는 각 lstm의 hidden state를 출력하는 인자이다.
# =Bidirectional(LSTM(250, return_sequences=True), input_shape=(text_max_words, EMB_DIM))(emb)

# #         SelfAttention(
# #             Sequential(
# #                 Linear(in_features=500, out_features=1, bias=True)
# #                 Tahh()
# #                 Dropout(0.3)f
# #                 Linear(in_features=500, out_features=1, bias=True)
# #                 Tanh()
# #                 Dropout(p=0.3)
# #             )
# #             Softmax()
# #         )
# #     )
# #     Linear(in_features=500, out_features=11, bias=True)    binary classification
# # )





# model.compile()

# word_vec = []
# for sent in tmp:
#     sub = []
#     for word in sent:
#         if(word in embeddings_index.keys()):
#             sub.append(embeddings_index[word])
#         else:
#             sub.append(np.random.uniform(-0.25,0.25,300)) ## used for OOV words
#     word_vec.append(sub)

# return np.array(word_vec)