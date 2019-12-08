import os
import sys
import numpy as np
import keras
import cv2
import argparse
import keras.backend as K
from keras.layers import *
from keras.optimizers import Adamax, RMSprop
from keras.models import Sequential, Model, load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator

def build_model(img_size=(64, 64), latent_dim=4):
    print("building model")
    assert img_size[0] == img_size[1] # images should be square

    img_input = keras.Input(shape=(img_size[0], img_size[1], 3), name='encoder_input')

    x = Conv2D(16, 3, activation='relu', padding='same')(img_input)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    shape_before_flatten = K.int_shape(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, name='codings')(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(np.prod(shape_before_flatten[1:]), activation='relu')(x)
    x = Reshape(shape_before_flatten[1:])(x)

    x = Conv2DTranspose(64, 3, padding='same', strides=(2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)

    x = Conv2DTranspose(32, 3, padding='same', strides=(2, 2))(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    x = Conv2DTranspose(16, 3, padding='same', strides=(2, 2))(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)

    x = Conv2D(3, 1, padding='same')(x)

    model = Model(img_input, x)

    model.summary()

    return model

def face_generator(face1_dir,
                   face2_dir,
                   batch_size=8,
                   target_size=(64, 64),
                   val_split=0.3,
                   subset='none'):

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


        yield (np.array(batch_y), np.array(batch_y))

def train_pipeline(epochs,
                   batch_size=8,
                   img_size=(64, 64),
                   val_split=0.3,
                   lr=1e-3,
                   loss='mse',
                   visual=False):

    model = build_model(img_size=img_size, latent_dim=4)


    optimizer = Adamax(lr=lr, decay=lr/(epochs))
    model.compile(loss=loss, optimizer=optimizer)

    face1_dir = os.path.join(os.getcwd(), "faces/face1/face")
    face2_dir = os.path.join(os.getcwd(), "faces/face2/face")

    val_size = int((len(os.listdir(face1_dir)) + len(os.listdir(face2_dir))) * val_split)
    train_size = (val_size / val_split) * (1 - val_split)

    print(train_size, val_size)

    train_gen = face_generator(face1_dir,
                               face2_dir,
                               subset='train',
                               target_size=img_size,
                               batch_size=batch_size,
                               val_split=val_split)

    val_gen = face_generator(face1_dir,
                             face2_dir,
                             subset='val',
                             val_split=val_split,
                             batch_size=batch_size,
                             target_size=img_size)

    if not visual:
        print("Fitting model")
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=train_size//batch_size, validation_data=val_gen, validation_steps=val_size//batch_size)
        print("Done")

        model.save(os.path.join(os.getcwd(), "src/dual_model.h5"))

    else:
        train_steps = (train_size // batch_size)
        val_steps = (val_size // batch_size)

        class VisualizeCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                test_img2 = cv2.imread(os.path.join(face2_dir, os.listdir(face2_dir)[500]), cv2.COLOR_BGR2RGB)
                test_img1 = cv2.imread(os.path.join(face1_dir, os.listdir(face1_dir)[500]), cv2.COLOR_BGR2RGB)
                test_img2 = cv2.resize(test_img2, img_size)
                test_img1 = cv2.resize(test_img1, img_size)
                self.test_img1 = np.array([test_img1 / 255.]).astype(np.float32)
                self.test_img2 = np.array([test_img2 / 255.]).astype(np.float32)

            def on_epoch_begin(self, epoch, logs={}):
                test_img1 = cv2.imread(os.path.join(face1_dir, os.listdir(face1_dir)[epoch + 1]), cv2.COLOR_BGR2RGB)
                test_img2 = cv2.imread(os.path.join(face2_dir, os.listdir(face2_dir)[epoch + 1]), cv2.COLOR_BGR2RGB)
                test_img1 = cv2.resize(test_img1, img_size)
                test_img2 = cv2.resize(test_img2, img_size)
                self.test_img1 = np.array([test_img1 / 255.]).astype(np.float32)
                self.test_img2 = np.array([test_img2 / 255.]).astype(np.float32)

            def on_epoch_end(self, epoch, logs={}):
                if epoch > 4:
                    if logs['loss'] - logs['val_loss'] < -0.05:
                        self.model.save(os.path.join(os.getcwd(), "src/dual_model.h5"))
                        print("not training well, model saved, exiting")
                        self.model.stop_training = True

            def on_batch_end(self, batch, logs={}):
                pred1 = self.model.predict(self.test_img1)[0]
                pred2 = self.model.predict(self.test_img2)[0]
                output_og1 = np.array(self.test_img1[0] * 255).astype('uint8')
                output_og2 = np.array(self.test_img2[0] * 255).astype('uint8')
                output_pred1 = pred1 - np.min(pred1)
                output_pred1 = output_pred1 / np.max(output_pred1)
                output_pred1 = np.array(output_pred1 * 255).astype('uint8')
                output_pred1 = cv2.cvtColor(output_pred1, cv2.COLOR_BGR2RGB)

                output_pred2 = pred2 - np.min(pred2)
                output_pred2 = output_pred2 / np.max(output_pred2)
                output_pred2 = np.array(output_pred2 * 255).astype('uint8')
                output_pred2 = cv2.cvtColor(output_pred2, cv2.COLOR_BGR2RGB)

                output2 = np.concatenate((output_og2, output_pred2), axis=1)
                output1 = np.concatenate((output_og1, output_pred1), axis=1)

                output = np.concatenate((output1, output2), axis=0)
                cv2.imshow("train image", output)
                cv2.waitKey(1)


        callbacks = [VisualizeCallback()]
        history = model.fit_generator(train_gen,
                                      epochs=epochs,
                                      steps_per_epoch=train_steps,
                                      validation_data=val_gen,
                                      validation_steps=val_steps,
                                      callbacks=callbacks)

        cv2.destroyAllWindows()
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
                    loss='mse',
                    visual=visualize)

    # build_model(img_size=(128, 128))
