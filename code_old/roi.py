import cv2
import numpy as np
import sys
import re

with open("./lib/list_cam_test.txt", "r") as f:
    list_cam = f.read().splitlines()

def main(argv):

    for vpath in list_cam:
        cap = cv2.VideoCapture(vpath + "vdo.avi")
        camid = int(*re.findall("c([\d]{3})", vpath)) 
        roi_mask = cv2.imread(vpath + "roi.jpg", cv2.IMREAD_GRAYSCALE)

        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        
        cv2.imwrite('./dump/c{:03d}_wroi.jpg'.format(camid), roi_frame)

if __name__ == "__main__":
    main(sys.argv[1:])
