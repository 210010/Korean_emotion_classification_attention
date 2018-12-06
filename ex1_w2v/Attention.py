from keras.layers import Activation, Concatenate, Permute, SpatialDropout1D, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, Input,Flatten,Embedding
from keras.models import Model
import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.regularizers import l2

# kernel_regularizer=regularizers.l2(0.0001)
# ,  W_regularizer=l2(0.0001)
class Attention:
    def __call__(self, inp,rate_drop_dense,l2_reg=0, combine=True, return_attention=True):
        # Expects inp to be of size (?, number of words, embedding dimension)
        # (6619,100,500)
        W_reg = l2(l2_reg)
        b_reg = l2(l2_reg)
        # self.W_const = 
        # self.b_const = 
        repeat_size = int(inp.shape[-1])
        
        # Map through 1 Layer MLP
        x_a = Dense(repeat_size, kernel_initializer = 'glorot_uniform', activation="tanh",name="tanh_mlp",W_regularizer=W_reg,b_regularizer=b_reg)(inp)
        x_a = Dropout(rate_drop_dense)(x_a)
        x_a = Dense(repeat_size, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp2",W_regularizer=W_reg,b_regularizer=b_reg)(x_a)
        x_a = Dropout(rate_drop_dense)(x_a)

        # Dot with word-level vector
        x_a = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context",W_regularizer=W_reg,b_regularizer=b_reg)(x_a)
        x_a = Dropout(rate_drop_dense)(x_a)
        # x_a is of shape (?,200,1), we flatten it to be (?,200)
        x_a = Flatten()(x_a) # x_a is of shape (6619,100,1), we flatten it to be (6619,100)
        att_out = Activation('softmax')(x_a) 
        
        # Clever trick to do elementwise multiplication of alpha_t with the correct h_t:
        # RepeatVector will blow it out to be (?,120, 200)
        # RepeatVector will blow it out to be (?,500, 100)
        # Then, Permute will swap it to (?,200,120) where each row (?,k,120) is a copy of a_t[k]
        # Then, Permute will swap it to (?,100,500) where each row (?,k,500) is a copy of a_t[k]
        # Then, Multiply performs elementwise multiplication to apply the same a_t to each
        # dimension of the respective word vector
        x_a2 = RepeatVector(repeat_size)(att_out)
        x_a2 = Permute([2,1])(x_a2)
        out = Multiply()([inp,x_a2])
        
        if combine:
        # Now we sum over the resulting word representations
            out = Lambda(lambda x : K.sum(x, axis=1), name='expectation_over_words')(out)
        
        if return_attention:
            out = (out, att_out)
                   
        return out