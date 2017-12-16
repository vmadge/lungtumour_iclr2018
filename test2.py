import numpy as np
import scipy.io as sio
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

def load_data():
    mat = sio.loadmat('dataset.mat')
    X = np.array(mat['files'])
    Y = np.array(mat['labels'])
    return X, Y

def shuffle(X,Y):
    shuffle = np.random.permutation(X.shape[0])
    X = X[shuffle,:]
    Y = Y[shuffle,:]
    return X,Y

def int2hc(Y):
    classes = np.unique(Y)
    HC = np.zeros((Y.shape[0], classes.shape[0]))
    for c in range(classes.shape[0]):
        HC[:,c] = (Y==c)*1
    return HC, classes

def hc2int(HC, classes):
    Y = [int(classes[i]) for i in np.argmax(HC, axis=1)]
    return np.array(Y)

def split_dataset(X, Y, ratio=0.8):
    Xr, Yr = X,Y #shuffle(X,Y)
    split = int(np.ceil(Xr.shape[0] * ratio))  # split int from ration
    Xt, Xv = np.vsplit(Xr, [split])
    Yt, Yv = Yr[:split], Yr[split:]
    return Xt, Yt, Xv, Yv

batch_size = 64
epochs = 20
img_rows, img_cols = 64, 64

[X, Y] = load_data()
Y, classes = int2hc(Y)
num_classes = len(classes)
[x_train, y_train, x_valid, y_valid] = split_dataset(X, Y, 0.8)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
x = x.reshape(x.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

fit = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid))
score = model.evaluate(x_valid, y_valid, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

y = model.predict(x)
y.dump('output/cnn.npy')

yn = np.argmax(y, axis=1)
o = classes[yn]

with open('output/cnn.csv', 'w') as f:
    f.write('Id,Label\n')
    id = 1
    for i in o:
        f.write(str(id)+','+str(i)+'\n')
        id += 1
