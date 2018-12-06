from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import *
# to visualize, and to make zero shape matrix
from attention_utils import get_activations, get_data_recurrent
from Attention import attention

INPUT_DIM = 300 #wordvec사이즈
TIME_STEPS = 280 #문장의 max길이

# # if True, the attention vector is shared across the input_dimensions where the attention is applied.
# SINGLE_ATTENTION_VECTOR = False
# APPLY_ATTENTION_BEFORE_LSTM = False

# # bi_lstm hidden_states(500,280) * attention_score  = (500,)
# def attention_3d_block(inputs):
#     # inputs = lstm layer
#     # inputs.shape = (batch_size,time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     #  이녀석의 기능은? transpose 한다.
#     a = Permute((2, 1))(inputs)
#     # lstm output 280이 time_step만큼있다. 하지만 bi_lstm으로 해야하므로 280*2,250(lstm_unit) 이 있어야 한다.
#     a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     # TODO attention score 짜는 알고리즘 새로 개편해야 한다. 
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul

# def model_attention_applied_after_lstm():
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     lstm_units = 250

#     lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
#     attention_mul = attention_3d_block(lstm_out)
#     attention_mul = Flatten()(attention_mul)
#     output = Dense(11, activation='sigmoid')(attention_mul)
#     model = Model(input=[inputs], output=output)
#     return model






# model setting

if __name__ == '__main__':
    lstm_shape = 250
    

    # Define the model
    inp = Input(shape=(text_max_words,))
    emb = Embedding(input_dim=vocab_size, output_dim=EMB_DIM,
                trainable=False, weights=[embedding_matrix], input_length=text_max_words)(inp)
    # max_features = vocab_size, maxlen=text_max_words, embed_size=EMB_DIM
    # emb = Embedding(input_dim=max_features, input_length = maxlen, output_dim=embed_size)(inp)
    x = SpatialDropout1D(0.1)(emb)
    x = Bidirectional(LSTM(lstm_shape, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x, attention = Attention()(x)
    x = Dense(11, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    attention_model = Model(inputs=inp, outputs=attention) # Model to print out the attention data
    model.summary()
    # verbose= ? , validation_split은 validation file로 변환시켜주어야 한다.
    
    # model.fit(X_t, y, validation_data=(x_val,y_val), epochs=3, verbose=1, batch_size=512)
    model.fit(X_t, y, validation_data=(x_val,y_val), epochs=50, verbose=1, batch_size=32)



# 테스트용 코드 
    def get_word_importances(text):
    lt = tokenizer.texts_to_sequences([text])
    x = pad_sequences(lt, maxlen=maxlen)
    p = model.predict(x)
    att = attention_model.predict(x)
    return p, [(reverse_token_map.get(word), importance) for word, importance in zip(x[0], att[0]) if word in reverse_token_map]
