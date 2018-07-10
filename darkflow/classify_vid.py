import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

SCREEN_WIDTH_RES = 1280
SCREEN_HEIGHT_RES = 720
SCREEN_LENGTH = 12.35
HAND_WIDTH = 3.15
FOCAL_LENGTH = 1750

def classify(step=25875):
    params = {
        'model': 'cfg/tiny-yolo-voc-1c.cfg',
        'load': step,
        'threshold': 0.62,
        'gpu': 1.0
    }

    tfnet = TFNet(params)
    colors = [tuple(266*np.random.rand(3)) for _ in range(10)]

    capture = cv2.VideoCapture(0)  # Gets default camera
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH_RES)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT_RES)

    while(1 > 0):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                hand_pixel_width = br[0] - tl[0]
                distance_ft = (HAND_WIDTH * FOCAL_LENGTH) / (12 * hand_pixel_width)

                if distance_ft > 13:  # Reduce noise, probably not a person on a webcam
                    break

                label = result['label']
                confidence = result['confidence']
                text = '{} {:.1f} ft.'.format(label, distance_ft)
                # text = '{} {:.1f} ft. : {:.0f}%'.format(label, distance_ft, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(
                    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def main():
    classify()

if __name__ == '__main__':
    main()
