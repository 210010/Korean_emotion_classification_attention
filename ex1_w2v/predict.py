# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from keras.models import model_from_json
from konlpy.tag import Okt as Twitter
import re
import json


# token2index에서는 '0'외에는 token들로 구성되어 있으며, index2vector값에서는 각 token들 index와, '0'의 index값과 마지막 index로 np.zeros(300)이 들어있다. 
class Predict():
    
    def __init__(self, sent="나는 오늘 화가 많이 났다."):
            # 1.데이터 준비하기
            self.sentence = sent
            # 2. 모델 불러오기
            json_file = open('model.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights('Attention_weight.h5')
            #이거 필요없을것 같은데
            self.index2vec = np.fromfile('index2vec.dat',dtype='float32')
#             self.index2vec = np.fromfile('index2vec.dat')
            self.max_sentence = 100
            #"토큰당 인덱스가 들어있는파일"
            with open('token2index.json','r') as dictionary_file:
                self.token2index = json.load(dictionary_file)
                
    def tokenizer(self):
        token = []
        twitter = Twitter()
        self.sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', self.sentence).strip().split()
        
        token.extend(twitter.pos(str(self.sentence), norm=True, stem=True))
        print(token)
        print(type(token))
        return token
    
    def t2index(self, t_list):
        index_list = []
        for token in t_list:
            if str(token).replace(" ", "") in self.token2index:
                index_list.append(self.token2index[str(token).replace(" ", "")])
            else:
                index_list.append(self.token2index['0'])
        #padding
        if(len(index_list)> self.max_sentence):
            return index_list[0:99]
        else:
            #token2index의key길이==zero가 있는 index이다. index2vec의 마지막 index와 len(token2index)같으면 된다. 
            #-1 지금 붙여논 상황
            index_list.extend([len(self.token2index)-1]*(self.max_sentence-len(index_list)))
            print(len(self.token2index))
            print(len(self.index2vec))
#         100개의 index 토큰으로 분류된 문장 return
        return index_list


from keras import backend as K

if __name__ == '__main__':
    sentence = input("문장을 입력하시오: ")
    predict = Predict(sentence)
    tok = predict.tokenizer()
    #토큰화 시킨 후 토큰화 된 단어를 token2index에 넣을떄, ()의 공백을 없애주어야 한다.
    #1.  in word_tmp_dict
    #2. 또한 x_test의 데이터타입과 token2index의 key데이터 타입이 같은지 확인해 주어야한다.
    # for i in token2index.keys(): print(type(i)), tuple(x_test) or str(x_test)
    x_test = predict.t2index(tok)
    #3. token2index파트에서, 전해진 tok 데이터 타입이 int가 되도록 해주어, index2vec의 index데이터 타입과 같도록 만들어준다. 
    #4. 넘파이화 시켜주어서, 이후 plot 찍어줄수 있도록한다.
    x_test = np.asarray([x_test])
#     print(x_test)
#     print(x_test.shape)
#     # 3. 모델 사용하기
#     #5. yhat이 [0.4,0.8,0.9.....]이런식으로 나오는데, threshold(0.5)를 넘기면 모두 1이 되도록 만들어주고 그게 아니면 0이되도록 변환시켜준다.
#     yhat = predict.model.predict(x_test,batch_size=1)
    yhat = predict.model.predict(x_test)
    yhat = yhat[0]
    
    #yhat중 1인 녀석 index를 뽑아낸다.
    columns = ["분노","기대","혐오","두려움","기쁨","사랑","낙관","비관","슬픔","놀라움","믿음"]
    y_index = list()
    for i,e in enumerate(yhat):
        y_index = [i for i, e in enumerate(yhat) if e > 0.3]
    for i in y_index:
        print("'{0}'가{1:.2f}% 발현되었습니다.".format(str(columns[i]),(yhat[i])*100))
    