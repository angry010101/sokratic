import numpy as np

from keras import Sequential, losses, optimizers, layers
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, RNN, Flatten, Dropout, \
    GlobalAveragePooling1D, Reshape, MaxPool1D
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

def getNum(x):
    table = dict({"0": 1,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6":1 ,"7": 0,"8": 2,"9": 1})
    res = 0
    for c in str(x):
        res += table[c]
    return res

n_samples = 1000
max_int = 10000
input_size = 10
chars = '0123456789'

X = list()
Y= list()
def pad_seq(x):
    while len(x)<input_size:
        x.append(1)
    return x


for i in range(0,n_samples):
    num = np.random.randint(0,max_int)
    numbers_list = list(str(num))
    numbers_list = [int(x) for x in numbers_list]
    numbers_list = pad_seq(numbers_list)
    X.append(numbers_list)
    Y.append(getNum(num))



print(X)
print(Y)

def normalize(x):
    return np.true_divide(np.array(x),9)

X = normalize(X)


print(X.shape)
X = np.expand_dims(X, axis=2)
print(X.shape)
# Explicitly set apart 10% for validation data that we never train over.
split_at = len(X) - len(Y) // 10
(x_train, x_val) = X[:split_at], X[split_at:]
(y_train, y_val) = Y[:split_at], Y[split_at:]

print(Y)
print(Y)


HIDDEN_SIZE = 128
LAYERS = 10
EPOCH = 50


RNN = layers.LSTM
KERNEL_SIZE = 2


BATCH_SIZE = 1

layers_count = 10

print('Build model...')
model = Sequential()
model.add(Conv1D(2,2,activation='relu',input_shape=(input_size,1)))
#for i in range(0,layers_count):
#    model.add(Conv1D(KERNEL_SIZE,HIDDEN_SIZE,activation='tanh'))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',
              loss=losses.MSE,metrics=['accuracy'])


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def toNum(x):
    r = []
    for i in x:
        r.append(getNum(i))
    return np.array(r)

import matplotlib.pyplot as plt
x_plot = np.arange(0,100)
y_plot = toNum(x_plot)
plt.plot(x_plot, y_plot, 'r')
plt.show()

for iteration in range(1, EPOCH+1):
    print()
    print('-' * EPOCH)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        preds = model.predict_classes(x_val, verbose=0)
        print("q ",x_val[i])
        if preds[i][0] == y_val[i]:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print("correct: ",y_val[i],"pred: ",preds[i][0])

