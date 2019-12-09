import numpy as np
import os
import sys
import argparse
import cv2
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import RMSprop, Adamax
import keras.backend as K

def load_original(src, input_name='encoder_input', output_name='codings'):
    model = load_model(src)
    shape_before_flatten = K.int_shape(model.get_layer("max_pooling2d_3").output)
    input_layer = model.get_layer(input_name).input
    output_layer = model.get_layer(output_name).output
    encoder = Model(input_layer, output_layer, name='encoder')
    encoder.trainable = False
    encoder.summary()
    return model, shape_before_flatten

def build_full(encoder, shape_before_flatten):
    full = Sequential(name='full_model')
    full.add(encoder)
    full.add(Dense(128, activation='relu'))
    full.add(Dense(np.prod(shape_before_flatten[1:]), activation='relu'))
    full.add(Reshape(shape_before_flatten[1:]))
    full.add(Conv2DTranspose(64, 3, padding='same', strides=(2, 2)))
    full.add(Conv2D(64, 3, padding='same', activation='relu'))
    full.add(Conv2D(64, 3, padding='same', activation='relu'))
    full.add(Conv2D(64, 3, padding='same', activation='relu'))
    full.add(Conv2DTranspose(32, 3, padding='same', strides=(2, 2)))
    full.add(Conv2D(32, 3, padding='same', activation='relu'))
    full.add(Conv2D(32, 3, padding='same', activation='relu'))
    full.add(Conv2D(32, 3, padding='same', activation='relu'))
    full.add(Conv2DTranspose(16, 3, padding='same', strides=(2, 2)))
    full.add(Conv2D(16, 3, padding='same', activation='relu'))
    full.add(Conv2D(16, 3, padding='same', activation='relu'))
    full.add(Conv2D(16, 3, padding='same', activation='relu'))
    full.add(Conv2D(3, 1, padding='same'))
    full.summary()
    return full

def model_builder(src, input_name='encoder_input', output_name='codings'):
    original, shape = load_original(src, input_name, output_name)

    new_model = build_full(original, shape)
    return new_model

def image_gen(src, img_size=(64, 64), val_split=0.3, batch_size=8, subset='none'):
    fnames = os.listdir(src)
    faces = []
    for fname in fnames:
        faces.append(os.path.join(src, fname))
    if subset == 'none':
        print()
    elif subset == 'train':
        faces = faces[int(val_split * len(faces)):]
    elif subset == 'val':
        faces = faces[:int(val_split * len(faces))]
    faces = np.array(faces)
    np.random.shuffle(faces)
    num_faces = faces.shape[0]
    print("Found {0} faces for {1}".format(num_faces, subset))
    batch_counter = 0
    while True:
        batch_paths = []
        if batch_counter + batch_size > num_faces:
            batch_paths = faces[batch_counter:]
            diff = batch_size - len(batch_paths)
            batch_paths = np.append(batch_paths, faces[:diff - 1])
            batch_counter = diff - 1
        else:
            batch_paths = faces[batch_counter: batch_counter + batch_size]
            batch_counter += batch_size
        batch_y = []
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, img_size)

            img = img / 255.

            batch_y.append(img)
        yield (np.array(batch_y), np.array(batch_y))

def train_pipeline(epochs, batch_size=8, img_size=(64, 64), val_split=0.3, lr=1e-3, loss='mse', visual=False):
    model_path = os.path.join(os.getcwd(), 'src/dual_model.h5')

    # full_model = model_builder(model_path)
    full_model, _ = load_original(model_path)

    train_dir = os.path.join(os.getcwd(), 'faces/face1/face')
    train_gen = image_gen(train_dir, img_size=img_size, val_split=val_split, batch_size=batch_size, subset='train')
    val_gen = image_gen(train_dir, img_size=img_size, val_split=val_split, batch_size=batch_size, subset='val')

    val_steps = int((len(os.listdir(train_dir)) * val_split) // batch_size)
    train_steps = (len(os.listdir(train_dir)) - int(val_steps * batch_size)) // batch_size

    optimizer = Adamax(lr=lr, decay=(epochs / (lr)))
    full_model.compile(loss=loss, optimizer=optimizer)

    class VisualizeCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            path = os.path.join(train_dir, os.listdir(train_dir)[500])
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = np.array([img / 255.]).astype(np.float32)
            self.test_img = img

        def on_epoch_begin(self, epoch, logs={}):
            path = os.path.join(train_dir, os.listdir(train_dir)[epoch + 1])
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = np.array([img / 255.]).astype(np.float32)
            self.test_img = img

        def on_batch_end(self, batch, logs={}):
            sample = self.test_img
            pred = self.model.predict(sample)[0]
            pred = pred - np.min(pred)
            pred = pred / np.max(pred)
            output = np.array(pred * 255, np.uint8)
            img = np.array(sample[0] * 255, np.uint8)

            final = np.concatenate((img, output), axis=1)

            cv2.imshow('training output', final)
            cv2.waitKey(1)

    callbacks = []
    if visual:
        callbacks.append(VisualizeCallback())

    history = full_model.fit_generator(train_gen,
                                  epochs=epochs,
                                  steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)

    full_model.save(os.path.join(os.getcwd(), 'src/face1_gen.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=15, help='epochs for training')
    parser.add_argument('-b', '--batch', type=int, default=8, help='batch size for training')
    parser.add_argument('--split', type=float, default=.3, help='validation split for data')
    parser.add_argument('-s', '--size', type=int, default='64', help='image size for training (make sure this is correct)')
    parser.add_argument('-r', '--rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('-v', '--visualize', type=int, default=0, help='show training process')
    args = parser.parse_args()
    visualize = False
    if args.visualize == 1:
        visualize = True

    train_pipeline(args.epochs,
                    batch_size=args.batch,
                    img_size=(args.size, args.size),
                    val_split=args.split,
                    lr=args.rate,
                    loss='mse',
                    visual=visualize)

    # model = model_builder(os.path.join(os.getcwd(), 'src/dual_model.h5'))
