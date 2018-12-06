import pandas as pd
import numpy as np
def checkAnomaly_x_y(x_data,y_data):
    trimed_x = list()
    trimed_y = list()
    # trimed_y = np.ndarray()
    for i,y in enumerate(y_data):
        if len(y) == 11:
            sent = list()
            for j,f in enumerate(y):
                try:
                    float(f)
                    sent.append(f)
                except ValueError:
                    #value가 float화 되지 못한다면 다음 문장으로 넘어간다.
                    sent = list()
                    break
            if sent:
                trimed_x.append(x_data[i])
                trimed_y.append(y_data[i])

    print(np.array(trimed_x).shape)
    print(np.array(trimed_y).shape)

    return np.array(trimed_x), np.array(trimed_y).astype('float32')
            
    
