# keras version
from keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout, TimeDistributed
from keras.layers import Dense, Input, Flatten, Concatenate
from keras import GaussianNoise
from keras.models import Sequential
ModelWrapper(
    # /home/minwookje/coding/cbaziotis/ntua-slp-semeval2018/modules/nn/models.py
    FeatureExtractor(
        Embed(
            Embedding(804871,310)
            Dropout(0.1)
            GaussianNoise(mean=0.0, stddev=0.2)
        )
        # /home/minwookje/coding/cbaziotis/ntua-slp-semeval2018/modules/nn/modules.py
        RNNEncoder(
            LSTM(310, 250, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
            Dropout(p=0.3)
        )
        SelfAttention(
            Sequential(
                Linear(in_features=500, out_features=1, bias=True)
                Tahh()
                Dropout(0.3)
                Linear(in_features=500, out_features=1, bias=True)
                Tanh()
                Dropout(p=0.3)
            )
            Softmax()
        )
    )
    Linear(in_features=500, out_features=11, bias=True)    
)

# Embedding
MAX_SEQUENCE_LENGTH = 310
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_train = Embedding(,Dropout=0.1,GaussianNoise=(mean = 0.0, stddev = 0.2))
embedded_sequences_train= embedding_layer_train(sequence_input)

# RNNEncoder
l_lstm = Bidirectional(LSTM(250,num_layers=2,return_sequences=True,dropout=0.3, recurrent_dropout=0.3))(embedded_sequences_train)

# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Attention
def attention():
    
    l_dense = Dense(500, input_shape=(500,), activation='tanh')(l_lstm)
    l_drop = Dropout(0.3)(l_dense)
    ll_dense = Dense(500, input_shape=(500,), activation='tanh')(l_drop)
    ll_drop = Dropout(0.3)(ll_dense)

    lll_dense = Dense(1, activation='tahnh')(ll_drop)
    lll_drop = Dropout(0.3)(lll_dense)
    llll_dense = Dense(1,activation='softmax')(lll_drop)

    # step2  (1) masking 단계

    # step2  (2) re-normalize 단계



# 곱 까지 완료한 lstm의 갯수만큼의 layer들을 모두 합쳐서 문장갯수 804871당 310dim -> 500인 lstm과 attention(804871,1)
# (804871,500) * (804871,1) 스칼라 곱
# multiply = multiply([l_lstm,ll_drop])
l_merge = multiply([l_lstm, llll_dense])
# 모든 (804871,500) - > (1,500)
cont_layer = Concatenate(all)


# relu 쓰나??
Dense(11, )(cont_layer)
    