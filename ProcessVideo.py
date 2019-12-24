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

def make_pred(clipped, model, refiner):
    # sample = np.expand_dims(clipped, axis=-1)
    sample = np.expand_dims(clipped, axis=0)
    sample = sample / 255.

    pred = model.predict(sample)

    pred = pred - np.min(pred)
    pred = pred / np.max(pred)
    refined = model.predict(pred)
    pred = np.array(pred[0] * 255, np.uint8)

    refined = refined - np.min(refined)
    refined = refined / np.max(refined)
    refined = np.array(refined[0] * 255, np.uint8)

    return refined, pred

def blend_img(pred, original, coords, bsize=0.05):
    (startX, startY, endX, endY) = coords

    img_size = (endX-startX, endY-startY)
    pred = cv2.resize(pred, img_size)
    og_clip = original[startY:endY, startX:endX]

    left_border_pred = pred[:, 0:int(bsize*img_size[0])]
    right_border_pred = pred[:, int(1.-bsize*img_size[0]):int(img_size[0])]
    top_border_pred = pred[int(1.-bsize*img_size[1]):int(img_size[1]), :]
    bottom_border_pred = pred[0:int(bsize*img_size[1]),:]

    left_border_true = og_clip[:, 0:int(bsize*img_size[0])]
    right_border_true = og_clip[:, int(1.-bsize*img_size[0]):int(img_size[0])]
    top_border_true = og_clip[int(1.-bsize*img_size[1]):int(img_size[1]), :]
    bottom_border_true = og_clip[0:int(bsize*img_size[1]),:]

    left_border_new = .5*left_border_true + .5*left_border_pred
    right_border_new = .5*right_border_true + .5*right_border_pred
    top_border_new = .5*top_border_true + .5*top_border_pred
    bottom_border_new = .5*bottom_border_true + .5*bottom_border_pred

    # pred[:, 0:int(.05*img_size[0])] = left_border_new
    # pred[:, int(.95*img_size[0]):int(img_size[0])] = right_border_new
    # pred[int(.95*img_size[1]):int(img_size[1]), :] = top_border_new
    # pred[0:int(.05*img_size[1]),:] = bottom_border_new

    pred[:, 0:int(bsize*img_size[0])] = left_border_true
    pred[:, int(1.-bsize*img_size[0]):int(img_size[0])] = right_border_true
    pred[int(1.-bsize*img_size[1]):int(img_size[1]), :] = top_border_true
    pred[0:int(bsize*img_size[1]),:] = bottom_border_true


    return pred



def stitch_new(src, original, coords):
    (startX, startY, endX, endY) = coords
    imsize = (endX-startX, endY-startY)

    src = cv2.resize(src, imsize)

    new = np.copy(original)

    new[startY:endY, startX:endX] = src

    return new

def video_pipeline(src, model_path, refiner_path, output_size=(640, 360), visualize=1):
    prototxt_path = os.path.join(os.getcwd(), 'extraction_models/deploy.prototxt')
    caffemodel_path = os.path.join(os.getcwd(), 'extraction_models/weights.caffemodel')
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    cap = cv2.VideoCapture(src)

    predictor = load_model(model_path)
    refiner = load_model(refiner_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, output_size)

        coords = face_coords(frame, model)

        clipped = clip_face(frame, coords, img_size=(64, 64))

        refined, pred = make_pred(clipped, predictor, refiner)

        # pred = blend_img(pred, frame, coords, bsize=0.05)

        # out_pred = stitch_new(pred, frame, coords)
        # out_refined = stitch_new(refined, frame, coords)

        final = np.concatenate((clipped, refined), axis=1)

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
    refiner_path = ''
    if args.model == 'a':
        refiner_path = os.path.join(os.getcwd(), 'src', 'ae_b.h5')
    else:
        refiner_path = os.path.join(os.getcwd(), 'src', 'ae_a.h5')

    video_pipeline(args.source, model_path, refiner_path, output_size=output_size, visualize=args.visualize)
