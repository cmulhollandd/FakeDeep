import os
import sys
import numpy as np
import keras
import cv2
import argparse
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import *
from keras.optimizers import Adamax, RMSprop
from keras.models import Sequential, Model, load_model, model_from_json

def SubpixelConv2D(input_shape, scale=4, name=None):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        import tensorflow as tf
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)

def build_encoder(img_size=(64, 64)):
    print("\nBuilding Encoder")
    img_input = keras.Input(shape=(img_size[0], img_size[1], 3), name='encoder_input')

    x = Conv2D(64, 5, padding='same', strides=(2, 2))(img_input)
    x = LeakyReLU()(x)
    x = Conv2D(128, 5, padding='same', strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same', strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same', strides=(2, 2))(x)
    x = LeakyReLU()(x)

    shape_before_flatten = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(np.prod(shape_before_flatten[1:]))(x)

    x = Reshape(shape_before_flatten[1:])(x)
    x = Conv2D(1024, 3, padding='same')(x)
    x = SubpixelConv2D(img_size, scale=2, name='encoder_output')(x)

    encoder = Model(img_input, x, name='encoder')
    print("Done\n")
    return encoder

def build_decoder(img_size=(64, 64), input_shape=(None, 0), model_name=None):
    print('\nBuilding Decoder')
    img_input = keras.Input(shape=input_shape, name='decoder_input')

    x = Conv2D(512, 3, padding='same')(img_input)
    x = LeakyReLU(.1)(x)
    x = SubpixelConv2D(img_size, scale=2)(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = LeakyReLU(.1)(x)
    x = SubpixelConv2D(img_size, scale=2)(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = LeakyReLU(.1)(x)
    x = SubpixelConv2D(img_size, scale=2)(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = LeakyReLU(.1)(x)
    # x = SubpixelConv2D(img_size, scale=2)(x)

    x = Conv2D(3, 5, padding='same', activation='sigmoid')(x)

    decoder = Model(img_input, x, name=model_name)
    print("Done\n")
    return decoder

def build_model(img_size=(64, 64)):
    print("building model")
    assert img_size[0] == img_size[1] # images should be square

    encoder = build_encoder(img_size=img_size)

    coding_shape = K.int_shape(encoder.get_layer('encoder_output').output)[1:]

    decoder_a = build_decoder(img_size=img_size, input_shape=coding_shape, model_name='decoder_a')
    decoder_b = build_decoder(img_size=img_size, input_shape=coding_shape, model_name='decoder_b')

    ae_a = Sequential(name='ae_a')
    ae_a.add(encoder)
    ae_a.add(decoder_a)

    ae_b = Sequential(name='ae_b')
    ae_b.add(encoder)
    ae_b.add(decoder_b)

    encoder.summary()
    decoder_a.summary()
    ae_a.summary()

    return (ae_a, ae_b)

class FaceGenerator():
    def __init__(self, src_dir1, src_dir2, rescale=1./255):
        face1_imgs = os.listdir(src_dir1)
        face2_imgs = os.listdir(src_dir2)

        for i in range(0, len(face1_imgs)):
            face1_imgs[i] = src_dir1 + "/" + face1_imgs[i]

        for i in range(0, len(face2_imgs)):
            face2_imgs[i] = src_dir2 + "/" + face2_imgs[i]

        face1_imgs = np.array(face1_imgs)
        face2_imgs = np.array(face2_imgs)

        np.random.shuffle(face1_imgs)
        np.random.shuffle(face2_imgs)

        self.face1_imgs = face1_imgs
        self.face2_imgs = face2_imgs
        self.rescale = rescale

        print(f"Found {face1_imgs.shape[0]} images for face1")
        print(f"Found {face2_imgs.shape[0]} images for face2")

    def flow(self, face, target_size=(64, 64), batch_size=8):
        faces = None
        if face == '1':
            faces = self.face1_imgs
        elif face == '2':
            faces = self.face2_imgs

        np.random.shuffle(faces)
        batch_index = 0
        num_imgs = faces.shape[0]
        while True:
            batch_faces = []
            if batch_index + batch_size >= num_imgs:
                batch_faces = faces[batch_index:]
                diff = batch_size - len(batch_faces)
                diff_faces = faces[:diff-1]
                batch_faces = np.concatenate((batch_faces, diff_faces), axis=0)
                batch_index = diff
            else:
                batch_faces = faces[batch_index:batch_index+batch_size]
                batch_index += batch_size

            batch_x = []
            for fname in batch_faces:
                img = cv2.imread(fname, cv2.COLOR_BGR2RGB)
                if img is None:
                    continue
                img = cv2.resize(img, target_size)
                img = img * self.rescale

                batch_x.append(img)

            batch_x = np.array(batch_x)
            yield batch_x


def train_pipeline(epochs,
                   batch_size=8,
                   img_size=(64, 64),
                   lr=1e-3,
                   loss='mse',
                   visual=False):

    ae_a, ae_b = build_model(img_size=img_size)
    optimizer = Adamax(lr=lr, decay=lr/(epochs))
    ae_a.compile(loss=loss, optimizer=optimizer)
    ae_b.compile(loss=loss, optimizer=optimizer)

    face1_dir = os.path.join(os.getcwd(), "faces/face1/face")
    face2_dir = os.path.join(os.getcwd(), "faces/face2/face")

    image_gen = FaceGenerator(face1_dir, face2_dir, rescale=1./255)
    face1_gen = image_gen.flow('1', target_size=img_size, batch_size=batch_size)
    face2_gen = image_gen.flow('2', target_size=img_size, batch_size=batch_size)

    face1_steps = image_gen.face1_imgs.shape[0] // batch_size
    face2_steps = image_gen.face2_imgs.shape[0] // batch_size

    def visualize_output(modela, modelb, facea, faceb):
        preda = modelb.predict(np.array([facea]))[0]
        predb = modela.predict(np.array([faceb]))[0]

        outputs = np.concatenate((preda, predb), axis=0)
        outputs = outputs - np.min(outputs)
        outputs = outputs / np.max(outputs)

        inputs = np.concatenate((facea, faceb), axis=0)

        final = np.concatenate((inputs, outputs), axis=1)

        cv2.imshow("training outputs", final)
        cv2.waitKey(1)

    print("\nFitting Models\n")
    total_batches = face1_steps * epochs
    history_a = []
    history_b = []
    for batch in range(1, total_batches):
        face1_imgs = next(face1_gen)
        face2_imgs = next(face2_gen)
        a_loss = ae_a.train_on_batch(face1_imgs, face1_imgs)
        b_loss = ae_b.train_on_batch(face2_imgs, face2_imgs)

        history_a.append(a_loss)
        history_b.append(b_loss)

        metrics = "batch %3f / %3f : ae_a loss: %.3f ae_b loss: %.3f \r"%(batch, total_batches, a_loss, b_loss)

        print(metrics, end='')

        if visual:
            visualize_output(ae_a, ae_b, face1_imgs[0], face2_imgs[0])

    current = os.path.dirname(__file__)
    ae_a.save(os.path.join(current, "src/ae_a.h5"))
    ae_b.save(os.path.join(current, "src/ae_b.h5"))

    epochs = range(0, len(history_a) + 1)

    plt.plot(epochs, history_a)
    plt.plot(epochs, history_b)
    plt.show()

    print("\nDone\n:)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=15, help='epochs for training')
    parser.add_argument('-b', '--batch', type=int, default=8, help='batch size for training')
    parser.add_argument('-s', '--size', type=int, default='64', help='image size for training (make sure this is correct)')
    parser.add_argument('-r', '--rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('-v', '--visualize', type=int, default=0, help='show training process (0=no 1=yes)')
    parser.add_argument('-l', '--loss', type=str, default='mae', help='loss function to use (mae, mse, binary_crossentropy)')
    args = parser.parse_args()
    visualize = False
    if args.visualize == 1:
        visualize = True
    train_pipeline(args.epochs,
                    batch_size=args.batch,
                    img_size=(args.size, args.size),
                    lr=args.rate,
                    loss=args.loss,
                    visual=visualize)

    # build_model(img_size=(64, 64))
