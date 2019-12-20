import os
import sys
import argparse
import numpy as np
import cv2
import keras
from keras.models import load_model

def face_coords(img, model):
    """returns coordinates of face in (x_min, y_min, x_max, y_max)"""
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    return (startX, startY, endX, endY)

def clip_face(img, coords, img_size=(64, 64)):
    (startX, startY, endX, endY) = coords

    clipped = img[startY:endY, startX:endX]
    clipped = cv2.resize(clipped, img_size)

    return clipped

def make_pred(clipped, model):
    # sample = np.expand_dims(clipped, axis=-1)
    sample = np.expand_dims(clipped, axis=0)
    sample = sample / 255.

    pred = model.predict(sample)[0]

    pred = pred - np.min(pred)
    pred = pred / np.max(pred)
    pred = np.array(pred * 255, np.uint8)

    return clipped, pred

def stitch_new(pred, original, coords):
    (startX, startY, endX, endY) = coords
    imsize = (endX-startX, endY-startY)

    pred = cv2.resize(pred, imsize)

    new = np.copy(original)

    new[startY:endY, startX:endX] = pred

    return new, original

def video_pipeline(src, model_path, output_size=(640, 360), visualize=1):
    prototxt_path = os.path.join(os.getcwd(), 'extraction_models/deploy.prototxt')
    caffemodel_path = os.path.join(os.getcwd(), 'extraction_models/weights.caffemodel')
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    cap = cv2.VideoCapture(src)

    predictor = load_model(model_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, output_size)

        coords = face_coords(frame, model)

        clipped = clip_face(frame, coords, img_size=(64, 64))

        _, pred = make_pred(clipped, predictor)

        out, original = stitch_new(pred, frame, coords)

        final = np.concatenate((original, out), axis=1)

        # Save frames
        if visualize == 1:
            cv2.imshow("final", final)
            cv2.waitKey(1)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, default="", help='video to process')
    parser.add_argument('--width', type=int, default=640, help='width for output video')
    parser.add_argument('--height', type=int, default=360, help='height for output video')
    parser.add_argument('-m', '--model', type=str, default="a", help='model to use for processing (a or b)')
    parser.add_argument('-v', '--visualize', type=int, default=1, help='show process (0=no 1=yes)')
    args = parser.parse_args()

    output_size = (args.width, args.height)
    model_path = os.path.join(os.getcwd(), 'src', 'ae_{0}.h5'.format(args.model))

    video_pipeline(args.source, model_path, output_size=output_size, visualize=args.visualize)
