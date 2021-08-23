from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(os.path.join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape(
        [-1] + list(crop_shape) + [3])
    labels = np.memmap(os.path.join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)
    plt.show()


def tfl_model():
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

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 32)
    spatial_layer(2, 64)
    spatial_layer(2, 96)

    model.add(Flatten())
    dense_bn_relu(242)
    model.add(Dense(2, activation='softmax'))
    return model


def graph_accuracy(history):
    # compare train vs val accuracy,
    # why is val_accuracy not as good as train accuracy? are we overfitting?
    epochs = history.history
    epochs['train_accuracy'] = epochs['accuracy']
    plt.figure(figsize=(10, 10))
    for k in ['train_accuracy', 'val_accuracy']:
        plt.plot(range(len(epochs[k])), epochs[k], label=k)
    plt.legend()
    plt.show()


def predict_and_evaluate(val):
    predictions = m.predict(val['images'])
    sbn.distplot(predictions[:, 0])
    predicted_label = np.argmax(predictions, axis=-1)
    print('accuracy:', np.mean(predicted_label == val['labels']))
    viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val)


data_dir = 'gtFine/'
datasets = {
    'val': load_tfl_data(os.path.join(data_dir, 'val')),
    'train': load_tfl_data(os.path.join(data_dir, 'train')),
}
train, val = datasets['train'], datasets['val']
if os.path.exists("model.h5"):
    m = load_model("model.h5")
else:
    # prepare our model
    m = tfl_model()
    m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    # train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting
    # the label of the images in the val dataset.
    history = m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=2)
    m.summary()
    graph_accuracy(history)
predict_and_evaluate(val)

# save model
# m.save("model.h5")


def save_model(m):
    """
    If you want to make sure that this model can be used on different operating systems and different
    versions of keras or tensorflow, this is the better way to save. For this project the simpler
    method above should work fine.
    """
    json_filename = 'model.json'
    h5_filename = 'weights.h5'
    # create a json with the model architecture
    model_json = m.to_json()
    # save the json to disk
    with open(json_filename, 'w') as f:
        f.write(model_json)
    # save the model's weights:
    m.save_weights(h5_filename)
    print(" ".join(["Model saved to", json_filename, h5_filename]))