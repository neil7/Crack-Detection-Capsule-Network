# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:10:00 2018

@author: Neil sharma
"""


import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap
#from keras.models import Sequential
#from keras.layers import Flatten
#from keras.layers import Dense
from keras import utils as np_utils
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')



def CapsNet(input_shape, n_class):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    # Layer 1a: Convolutional layer for detecting the edges from the image
    # Layer 1b: Relu layer for adding a non-linearity
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    # Layer 2a: 6*6*32
    # Layer 2b: Squashing
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16,
                             name='digitcaps')(primarycaps)

    
    #Flattening
    A = layers.Flatten()(digitcaps)

    #Full connection
    B = layers.Dense(32, input_shape = (2,), activation = 'relu')(A)
    C = layers.Dense(n_class, activation = 'relu')(B)
    D = layers.Softmax()(C)
    

    
    # Models for training and evaluation (prediction)
    train_model = models.Model([x], D)
    
    
    return train_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train1(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

#    # callbacks
#    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
#    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
#                               batch_size=args.batch_size, histogram_freq=int(args.debug))
#    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
#                                           save_best_only=True, save_weights_only=True, verbose=1)
#    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  loss_weights=[args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    batch_size = 50
    epochs = 10
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]])
    

    """# Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    validation_data = [[x_test, y_test], [y_test, x_test]]
    
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data = validation_data,
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#"""

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

#    from utils import plot_log
#    plot_log(args.save_dir + '/log.csv', show=True)

    return model




def load_data():
    x_train = np.load('Crack.npy')
    x_train = x_train.reshape(-1,32,32,1)
    label = np.ones((40000,),dtype = int)
    label[0:20000] = 0
    label[20000:40001] = 1
    
    n_classes = 2
    
    train_data = [x_train, label]
    (x, y) = (train_data[0], train_data[1])
    y = np_utils.to_categorical(y, n_classes)
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4 )
#    (x, y) = (train_data[0], train_data[1])
#    Y_train = np_utils.to_categorical(y_train, n_classes)
#    (x_test, y_test) = (train_data[0], train_data[1])
#    Y_test = np_utils.to_categorical(y_test, n_classes)
    
    return (X_train, y_train), (X_test, y_test)



if __name__ == "__main__":
    import os
    import argparse
#    from keras.preprocessing.image import ImageDataGenerator
#    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (X_train, y_train), (X_test, y_test) = load_data()
    input_shape = (32, 32, 1)
    batch_size = 100
    epochs = 50
    n_class = 2
    validation_data = (X_test, y_test)
    # define model
    model = CapsNet(input_shape=input_shape, n_class=n_class)
    model.summary()
    
    model.compile(optimizer='adam',
                  loss = 'mean_squared_error',
                  metrics=['accuracy'])
    
    
    model.fit(X_train, y_train, batch_size = batch_size,  epochs=epochs,
              validation_data=validation_data)
    
    model.save('CapsNet_Crack.h5')
    print("Saved model to dsik")

#    from keras.preprocessing import image
#    test_image = image.load_img('img.jpg', target_size = (28, 28))
#    test_image = image.img_to_array(test_image)
#    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict_generator(X_test, y_test)
    print (model.evaluate(X_test, y_test))
    #filter = dim