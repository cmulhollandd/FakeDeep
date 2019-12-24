import os
import sys
import numpy as np
import cv2
import argparse

prototxt_path = os.path.join(os.getcwd(), 'extraction_models/deploy.prototxt')
caffemodel_path = os.path.join(os.getcwd(), 'extraction_models/weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def extract_face(img, output_size=(64, 64)):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    frame = None
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        if (confidence > 0.5):
            frame = img[startY:endY, startX:endX]
            frame = cv2.resize(frame, output_size)

    return frame

def loop_video(src, out, target_size=(64, 64), max_frames=None, show_frame=False):
    cap = cv2.VideoCapture(src)
    counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame found, exiting")
            return

        frame = cv2.resize(frame, (640, 360))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        extracted = extract_face(img, output_size=target_size)

        if extracted is None:
            continue

        output = cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(out, "{0}.jpg".format(counter)), output)
        print("{0}\r".format(counter), end='')
        counter += 1
        if show_frame:
            extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
            cv2.imshow("output", extracted)
            cv2.waitKey(1)

        if max_frames is not None:
            if counter >= max_frames:
                print("{0} frames processed, exiting".format(max_frames))
                return

    cv2.destroyAllWindows()
    return counter


def process_videos(input_paths, target_size=(64, 64), max_frames=None, show_frame=False):
    for i in range(0, len(input_paths)):
        output_path = os.getcwd()
        output_path = os.path.join(output_path, "faces")
        output_path = os.path.join(output_path, "face{0}".format(i + 1))
        output_path = os.path.join(output_path, 'face')
        try:
            os.mkdir(output_path)
        except:
            print("output {0} already exists, continuing".format(output_path))

        loop_video(input_paths[i], output_path, target_size=target_size, max_frames=max_frames, show_frame=show_frame)

def format_dirs():
    current = os.getcwd()

    faces = os.path.join(current, 'faces2')
    face1 = os.path.join(faces, "face1")
    face2 = os.path.join(faces, "face2")

    try:
        os.mkdir(faces)
        os.mkdir(face1)
        os.mkdir(face2)
        os.mkdir(os.path.join(face2, "face"))
        os.mkdir(os.path.join(face1, "face"))
    except:
        print("output directories already exist, continuing")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, nargs=2, default='', help='paths to source videos')
    parser.add_argument('-s', '--size', type=int, default=64, help='target size for extracted images')
    parser.add_argument('-f', '--frames', type=int, default=None, help='max number of frames to extract faces from')
    parser.add_argument('-v', '--visualize', type=int, default=0, help='visualize processing (0=no 1=yes)')

    args = parser.parse_args()

    print(args)

    visualize = False
    if args.visualize == 1:
        visualize = True


    format_dirs()

    process_videos(args.source,
                    target_size=(args.size, args.size),
                    max_frames=args.frames,
                    show_frame=visualize)


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
