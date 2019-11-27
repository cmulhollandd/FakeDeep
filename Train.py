import os
import sys
import numpy as np
import keras
import cv2
import argparse
from keras.layers import *
from keras.optimizers import Adamax, RMSprop
from keras.models import Sequential, Model, load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator

def build_model(img_size=(64, 64), block_num=10):
    print("building model")
    if img_size != (64, 64):
        input_layer = keras.Input(shape=(img_size[0], img_size[1], 3), name='input_layer')
        x = Conv2D(64, 3, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 1, padding='same', name='codings')(x)

        x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(16, 3, padding='same')(x)
        output = Conv2D(3, 3, padding='same')(x)

        model = Model(input_layer, output)

        model.summary()

        return model

    else:
        json_path = os.path.join(os.getcwd(), "src")
        json_path = os.path.join(json_path, 'dual_model.json')

        model_architecture = ''
        with open(json_path) as file:
            model_architecture = file.read()

        model = model_from_json(model_architecture)

        model.summary()

        return model

def face_generator(face1_dir, face2_dir, batch_size=8, target_size=(64, 64), val_split=0.3, subset='none'):
    face1_imgs = os.listdir(face1_dir)
    face2_imgs = os.listdir(face2_dir)

    faces = []

    for path in face1_imgs:
        faces.append(os.path.join(face1_dir, path))
    for path in face2_imgs:
        faces.append(os.path.join(face2_dir, path))


    paths = np.array(faces)
    np.random.shuffle(paths)

    faces = paths.tolist()

    if subset == 'none':
        print()
    elif subset == 'train':
        faces = faces[int(val_split * len(faces)):]
    elif subset == 'val':
        faces = faces[:int(val_split * len(faces))]

    paths = np.array(faces)
    np.random.shuffle(paths)

    num_faces = paths.shape[0]

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

            img = cv2.resize(img, target_size)

            img = img / 255.

            batch_y.append(img)

        yield (np.array([batch_y, batch_y]))

def train_pipeline(epochs, batch_size=8, img_size=(64, 64), val_split=0.3, lr=1e-2, loss='mse', visual=False):
    model = None
    if img_size != (64, 64):
        model = build_model(img_size)
    else:
        model = build_model()

    optimizer = Adamax(lr=lr, decay=lr/(epochs * .5))
    model.compile(loss=loss, optimizer=optimizer)

    face1_dir = os.path.join(os.getcwd(), "faces/face1/face")
    face2_dir = os.path.join(os.getcwd(), "faces/face2/face")

    val_size = int((len(os.listdir(face1_dir)) + len(os.listdir(face2_dir))) * val_split)
    train_size = (val_size / val_split) * (1 - val_split)

    train_gen = face_generator(face1_dir, face2_dir, subset='train', target_size=img_size, batch_size=batch_size, val_split=val_split)
    val_gen = face_generator(face1_dir, face2_dir, subset='val', val_split=val_split, batch_size=batch_size, target_size=img_size)

    if not visual:
        print("Fitting model")
        history = model.fit_generator(train_gen, steps_per_epoch=train_size//batch_size, validation_data=val_gen, validation_steps=val_size//batch_size)
        print("Done")

        model.save(os.path.join(os.getcwd(), "src/dual_model.h5"))

    else:
        train_steps = train_size // batch_size
        val_steps = val_size // batch_size


        print("Fitting model")
        for epoch in range(0, epochs):
            for train in range(0, int(train_steps)):
                batch_x, batch_y = next(train_gen)
                model.train_on_batch(batch_x, batch_y)
                test = batch_x[-1]
                sample = np.array([test])
                pred = model.predict(sample)[0]
                pred = pred - np.min(pred)
                pred = pred / np.max(pred)
                pred = np.array(pred * 255).astype('int8')
                out = np.concatenate((np.array(test * 255).astype('int8'), pred), axis=1)
                cv2.imshow('train image', out)
                cv2.waitKey(1)
            for val in range(0, int(val_steps)):
                batch_x, batch_y = next(val_gen)
                model.train_on_batch(batch_x, batch_y)
                test = batch_x[-1]
                sample = np.array([test])
                pred = model.predict(sample)[0]
                pred = pred - np.min(pred)
                pred = pred / np.max(pred)
                pred = np.array(pred * 255).astype('int8')
                out = np.concatenate((np.array(test * 255).astype('int8'), pred), axis=1)
                cv2.imshow('train image', out)
                cv2.waitKey(1)

        print("Done")

        model.save(os.path.join(os.getcwd(), "src/dual_model.h5"))

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
                    visual=visualize)
