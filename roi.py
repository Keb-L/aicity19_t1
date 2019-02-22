import cv2
import numpy as np
import sys

with open("./lib/list_cam.txt", "r") as f:
    list_cam = f.read().splitlines()

def main(argv):

    vpath = list_cam[14]
    cap = cv2.VideoCapture(vpath + "vdo.avi")

    roi_mask = cv2.imread(vpath + "roi.jpg", cv2.IMREAD_GRAYSCALE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        compute_gray_distance(roi_frame)



def compute_gray_distance(image):
    # Compute gray-scale image (average RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    err_img = image.copy()

    cv2.imshow("orig", image)
    cv2.imshow("gray", gray)

    err_img[:, :, 0] -= gray
    err_img[:, :, 1] -= gray
    err_img[:, :, 2] -= gray

    cv2.waitKey(0)
    print()



if __name__ == "__main__":
    main(sys.argv[1:])
