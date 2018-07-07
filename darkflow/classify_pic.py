'''
Created on May 19th, 2018
author: Julian Weisbord
sources:
description: Loads the object detecotr model at a specific step and detect a series of
                images.
'''

import os
import sys
import cv2
import math
import tensorflow
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt


VERBOSE = True


def classify(data, step=1625):
    params = {
        'model': 'cfg/tiny-yolo-voc-1c_man.cfg',
        'load': step,
        'threshold': 0.1,
        'gpu': 1.0
    }
    predictions = {}
    tfnet = TFNet(params)
    for img in os.listdir(data):
        pth = str(img)
        # print("path for image: ", img)
        img = data + '/' + img
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = tfnet.return_predict(img)

        if VERBOSE:
            print(result)
        if not result:
            continue
        tl_corner = (result[0]['topleft']['x'], result[0]['topleft']['y'])
        br_corner = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
        if VERBOSE:
            print(tl_corner)
        label = result[0]['label']

        if VERBOSE:
            # Make a green bounding box around the image
            img = cv2.rectangle(img, tl_corner, br_corner, (0, 255, 0), 2)
            text_corner = (tl_corner[0] + 30, tl_corner[1] + 30)
            # Add text to box
            img = cv2.putText(img, label, text_corner, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            plt.rcParams["figure.figsize"] = (15, 15)

            plt.imshow(img)
            plt.show()
        predictions[pth] = [label, result[0]['confidence'], tl_corner, br_corner]

    print(predictions)
    return predictions
def main():
    if len(sys.argv) !=2:
        print("Incorrect command line args")
        exit()
    classify(sys.argv[1])

if __name__ == '__main__':
    main()
