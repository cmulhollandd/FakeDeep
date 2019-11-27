import os
import sys
import keras
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import ExtractFaces
import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=str, nargs=2, default='', help='paths to source videos')
    parser.add_argument('-s', '--size', type=int, default=64, help='target size for extraced images')
    parser.add_argument('-v', '--visualize', type=int, default=0, help='visualize extraction and training (0=no 1=yes)')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='number of epochs to train for')
    parser.add_argument('-f', '--frames', type=int, default=None, help='max number of images to extract from source videos')
    parser.add_argument('-b', '--batch', type=int, default=8, help='batch size for training')

    args = parser.parse_args()

    print(args)

    visualize = False

    if args.visualize == 1:
        visualize = True

    ExtractFaces.format_dirs()
    ExtractFaces.process_videos(args.videos,
                                target_size=(args.size, args.size),
                                max_frames=args.frames,
                                show_frame=visualize)



    Train.train_pipeline(args.epochs,
                        batch_size=args.batch,
                        img_size=(args.size, args.size),
                        val_split=.2,
                        lr=1e-3,
                        visual=visualize)

    print("Done :)")


if __name__ == "__main__":
    main()
