import numpy as np
from keras import Input, Sequential
from keras.layers import LSTM, Dense, Conv1D, Flatten, Activation, BatchNormalization, MaxPooling2D, Dropout, \
    MaxPooling1D, GlobalAveragePooling1D, Embedding, TimeDistributed, RepeatVector
from keras.legacy import layers
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import seaborn as sns

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)



def getNum(x):
    table = dict({"0": 1,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6":1 ,"7": 0,"8": 2,"9": 1})
    res = 0
    for c in str(x):
        res += table[c]
    return res



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


n_samples = 10000
max_int = 9999999999
input_size = 10
chars = '0123456789'
max_count = 21

table_input = CharacterTable(chars)
table_output = CharacterTable(chars)
seed = 0
X = list()
Y= list()

def getRandomInt():
    s = ""
    for i in range(0,input_size):
        s += str(np.random.randint(0,10))
    return s

def pad_seq(x):
    while len(x)<input_size:
        x.append('1')
    return x
def create_table(Y):
    Y_temp = []
    for i in Y:
        x = np.zeros(max_count)
        x[i] = 1
        Y_temp.append(x)
    return Y_temp

for i in range(0,n_samples):
    num = getRandomInt()
    numbers_list = list(str(num))
    numbers_list = [x for x in numbers_list]
    numbers_list = pad_seq(numbers_list)
    X.append(numbers_list)
    Y.append(getNum(str(num)))



def normalize(x):
    return np.true_divide(np.array(x),9)



#print(X)
#X = normalize(X)

print(X)

#Y = to_categorical(Y,  num_classes=20)

#Y = create_table(Y)
Y = to_categorical(Y,  num_classes=max_count)
print(Y)

x = np.zeros((len(X), input_size, len(chars)), dtype=np.bool)
y = Y
#y = np.zeros((len(Y), max_count, len(chars)), dtype=np.bool)

for i, sentence in enumerate(X):
    x[i] = table_input.encode(sentence, input_size)
#for i, sentence in enumerate(Y):
#    y[i] = table_output.encode(sentence, max_count)

print(x.shape)
split_at_train_test = len(X) - len(Y) // 20
#(x_train, x_test) = x[:split_at_train_test], x[split_at_train_test :]

#(y_train, y_test) = y[:split_at_train_test], y[split_at_train_test :]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=43,shuffle=True)

split_at_train_val = len(x_train) - len(y_train) // 10
(x_train,x_val) = x_train[:split_at_train_val],x_train[split_at_train_val:]
(y_train,y_val) = y_train[:split_at_train_val],y_train[split_at_train_val:]
# create the model

def getNumberPlot(x):
    y_plot = list()
    for i in x:
        y_plot.append(getNum(i))
    return y_plot

import matplotlib.pyplot as plt

x_plot = np.arange(0,99999)
y_plot = getNumberPlot(x_plot)
plt.plot(x_plot, y_plot, 'r')
plt.show()

print("MEAN X: ",np.mean(x_plot)," STD: ", np.std(x_plot))


def calcDistribution():
    sns.set(style="darkgrid")
    titanic = sns.load_dataset("titanic")
    ax = sns.countplot(x="class", data=titanic)
    sns.countplot(y_plot)
    plt.show()


calcDistribution()

print("MEAN Y: ",np.mean(y_plot)," STD: ", np.std(y_plot))



RNN = LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
embedding_vecor_length = 32
print(x.shape)
print('Build model...')
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True, clipvalue=1.)
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
#model.add(Embedding(input_size, embedding_vecor_length, input_length=max_int))
model.add(RNN(HIDDEN_SIZE, input_shape=(input_size, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(RepeatVector(max_count))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(Flatten())

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
#model.add(TimeDistributed(Dense(20, activation='softmax')))
model.add(Dense(max_count, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()


def arrtolabel(x):
    for i,v in enumerate(x):
        if v==1:
            return i
    return 0
REVERSE = False
print(x_train.shape)
print(y_train.shape)
cvscores = []
from keras.callbacks import History
history = History()

history_acc= list()
history_val_acc= list()
history_loss= list()
history_val_loss= list()

#model.load_weights('weightsfile.h5')
for iteration in range(1, 50):
    #continue
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,validation_data=(x_val,y_val),callbacks=[history])
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = table_input.decode(rowx[0])
        #correct = table_output.decode(rowy[0])
        correct = arrtolabel(rowy[0])
        guess = preds[0]
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct==guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    history_acc.append(history.history['acc'])
    history_val_acc.append(history.history['val_acc'])
    history_loss.append(history.history['loss'])
    history_val_loss.append(history.history['val_loss'])

print(history.history.keys())
# summarize history for accuracy
plt.plot(history_acc)
plt.plot(history_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_loss)
plt.plot(history_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('weightsfile.h5')
i = input("I: ")
while (i!=-1):
    r = getNum(str(i))
    arr = list()
    arr1 = pad_seq(list(str(i)))
    arr.append(table_input.encode(arr1,input_size))
    arr=np.array(arr)
    pred = model.predict(arr)
    pred_class = model.predict_classes(arr)
    pred_proba = model.predict_proba(arr)
    print("INT: ",i,"ACTUAL", r, " PREDICTED: ", pred_class, pred, " PREDICTED PROBA: ",pred_proba)
    i = input("I: ")

