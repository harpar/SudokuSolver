import sys
import tensorflow as tf
from helper import getModel

def trainModel():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    print('Number of images in training set', x_train.shape[0])

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    model = getModel(input_shape)
    model.fit(x=x_train,y=y_train, epochs=10)

    model.save_weights("./model")

def testModel(model_path="./model"):
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    input_shape = (28, 28, 1)
    model = getModel(input_shape)
    model.load_weights(model_path)

    # Making sure that the values are float so that we can get decimal points after division
    x_test = x_test.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_test /= 255

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    print(f"Result: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    if sys.argv[1] == "train":
        trainModel()
    elif sys.argv[1] == "test":
        testModel(sys.argv[2] if len(sys.argv) >= 3 else "./model")