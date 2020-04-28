import cv2 as cv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def showImage(image):
    cv.imshow("Sudoku Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getModel(input_shape):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

def classify(img):
    input_shape = (28, 28, 1)
    model = getModel(input_shape)
    model.load_weights("./model")

    prediction = model.predict(img)

    return prediction.argmax()

def getBoard(imgs):
    res = [['.'] * 9 for _ in range(9)]
    for (x, y), img in imgs.items():
        res[y][x] = str(classify(img))

    return res