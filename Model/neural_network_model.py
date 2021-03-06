import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    """
    function which load the dataset
    :param data_dir: path to the dataset (data.bin and labels.bin)
    :param crop_shape: shape of images
    :return: dictionary with two keys: 'images' for images np.array and 'labels' for labels np.array.
    """
    images = np.memmap(os.path.join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape(
        [-1] + list(crop_shape) + [3])
    labels = np.memmap(os.path.join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    """
    visualize the data using matplotlib window
    :param images: np.array of images
    :param labels: np.array of labels
    :param predictions: np.array of floats (between 0 to 1) of the neural network estimates if its a TFL or not
    :param num: how much to show in window
    :param labels2name: text to show in window, as default its No TFL / Yes TFL
    :return:
    """
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]),
                      gridspec_kw={'wspace': 0.05}, squeeze=False, sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)
    plt.show()


def tfl_model():
    """
    creates a tfl neural network Model
    :return: the instance which created in this function
    """
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(5, 5), input_shape=input_shape)
    conv_bn_relu(32, kernel_size=(5, 5))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(rate=0.25))
    spatial_layer(2, 64)
    spatial_layer(1, 128)
    model.add(Flatten())
    dense_bn_relu(256)
    model.add(Dropout(rate=0.25))
    model.add(Dense(2, activation='softmax'))
    return model


def graph_accuracy(history):
    """
    shows accuracy graph during the neural network training
    :param history: the history of the training is stored here
    :return:
    """
    # compare train vs val accuracy,
    # why is val_accuracy not as good as train accuracy? are we overfitting?
    epochs = history.history
    epochs['train_accuracy'] = epochs['accuracy']
    plt.figure(figsize=(10, 10))
    for k in ['train_accuracy', 'val_accuracy']:
        plt.plot(range(len(epochs[k])), epochs[k], label=k)
    plt.legend()
    plt.show()


def find_tfl(model, image,red_lights,green_lights):
    # # padding with 40 pixels
       zeroes = np.zeros((len(image) + 81, len(image[0]) + 81, 3))
       zeroes[41:image.shape[0] + 41, 41:image.shape[1] + 41] = image
       padded_image = zeroes.astype(dtype=np.uint8)

       cropped_red_images=[]
       cropped_green_images=[]

       for red_x,red_y in red_lights:
                   cropped = padded_image[red_y:red_y + 81, red_x:red_x + 81,:]
                   cropped_red_images.append(cropped.tolist())

       for green_x,green_y in green_lights:
                   cropped = padded_image[green_y:green_y + 81, green_x:green_x + 81,:]
                   cropped_green_images.append(cropped.tolist())

       predicted_red_label, predicted_green_label = predict(model,cropped_red_images,cropped_green_images)

       red_lights = [red_lights[i] for i in range(len(red_lights)) if predicted_red_label[i]]
       green_lights = [green_lights[i] for i in range(len(green_lights)) if predicted_green_label[i]]

       return red_lights,green_lights


def predict(model,cropped_red_images,cropped_green_images):
    '''

    :param model:
    :param cropped_red_images:
    :param cropped_green_images:
    :return: predicted result of tfl points
    '''
    predicted_red_label=[]
    predicted_green_label=[]

    if cropped_red_images:
        predictions_red = model.predict(cropped_red_images)
        predicted_red_label = np.argmax(predictions_red, axis=-1)

    if cropped_green_images:
        predictions_green = model.predict(cropped_green_images)
        predicted_green_label = np.argmax(predictions_green, axis=-1)

    return predicted_red_label,predicted_green_label


def predict_and_evaluate(model, val):
    """
    activate the images as input for the Model,
    the Model predict the input if its tfl or no for each image
    shows the results of the prediction facing the GT
    and shows graph of evaluation
    :param model: the tfl Model
    :param val: the dataset of validation
    :return:
    """
    predictions = model.predict(val['images'])
    sbn.distplot(predictions[:, 0])
    predicted_label = np.argmax(predictions, axis=-1)
    print('accuracy:', np.mean(predicted_label == val['labels']))
    viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val)
    return predictions,predictions

def save_model(model):
    """
    If you want to make sure that this Model can be used on different operating systems and different
    versions of keras or tensorflow, this is the better way to save. For this project the simpler
    method above should work fine.
    """
    json_filename = 'Model.json'
    h5_filename = 'weights.h5'
    # create a json with the Model architecture
    model_json = model.to_json()
    # save the json to disk
    with open(json_filename, 'w') as f:
        f.write(model_json)
    # save the Model's weights:
    model.save_weights(h5_filename)
    print(" ".join(["Model saved to", json_filename, h5_filename]))


def init():
    data_dir = '../Data/gtFine/'
    datasets = {
        'val': load_tfl_data(os.path.join(data_dir, 'val')),
        'train': load_tfl_data(os.path.join(data_dir, 'train')),
    }
    train, val = datasets['train'], datasets['val']

    if os.path.exists("../Data/Resource/model.h5"):
        m = load_model("../Data/Resource/model.h5")
    else:
        # prepare our Model
        m = tfl_model()
        m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
        # train it, the Model uses the 'train' dataset for learning.
        # We evaluate the "goodness" of the Model, by predicting
        # the label of the images in the val dataset.
        history = m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=4)
        #m.summary()
        graph_accuracy(history)
        # save Model
        m.save("Model.h5")

    return m


init()

