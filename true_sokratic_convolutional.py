import numpy as np
from keras import Input, Sequential
from keras.layers import LSTM, Dense, Conv1D, Flatten, Activation, BatchNormalization, MaxPooling2D, Dropout, \
    MaxPooling1D, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score


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
max_count = 20
seed = 0
X = list()
Y= list()
def pad_seq(x):
    while len(x)<input_size:
        x.append(1)
    return x
def create_table(Y):
    Y_temp = []
    for i in Y:
        x = np.zeros(max_count)
        x[i] = 1
        Y_temp.append(x)
    return Y_temp

for i in range(0,n_samples):
    num = np.random.randint(0,max_int)
    numbers_list = list(str(num))
    numbers_list = [int(x) for x in numbers_list]
    numbers_list = pad_seq(numbers_list)
    X.append(numbers_list)
    Y.append(getNum(num))



def normalize(x):
    return np.true_divide(np.array(x),9)


print(X)
X = normalize(X)
print(X)
Y = to_categorical(Y,  num_classes=20)

#Y = create_table(Y)
print(Y)

X = np.expand_dims(X, axis=2)
split_at = len(X) - len(Y) // 5
(x_train, x_val) = X[:split_at], X[split_at:]
(y_train, y_val) = Y[:split_at], Y[split_at:]

num_encoder_tokens = len(chars)
latent_dim = 256
num_decoder_tokens = len(chars)


def baseline_model():
    model = Sequential()
    #model.add(Conv1D(filters=64, kernel_size=10, strides=10,
    #                 input_shape=(input_size, 1), kernel_initializer='uniform',
    #                 activation='relu'))
    model.add(Activation("relu"))
    model.add(Conv1D(latent_dim, 1, activation='relu', input_shape=(input_size, 1)))
    model.add(Conv1D(latent_dim, 1, activation='relu'))
    model.add(Conv1D(latent_dim, 1, activation='relu'))
    model.add(Conv1D(latent_dim, 1, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(latent_dim, 1, activation='relu'))
    model.add(Conv1D(latent_dim, 1, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(20, activation='relu'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=15, verbose=2)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
estimator.fit(x_train, y_train, epochs=300, batch_size=15)

results = cross_val_score(estimator, x_val, y_val, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

i=1
while (i!=0):
    i = input("ENTER NUMBER: ")
    arr = np.ones(input_size)
    for e,c in enumerate(i):
        arr[e] = c

    print("INPUT ",arr)
    arr = normalize([arr])
    arr = np.expand_dims(arr, axis=2)
    print("INPUT ",arr)
    res = getNum(i)
    r = np.zeros(max_count)
    r[res] = 1
    print("EXPECTED",r)
    pred = estimator.predict(arr)
    pred_ = estimator.predict_proba(arr)
    print("PREDICTED",pred,pred_)