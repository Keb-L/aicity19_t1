"""

AICity 2019
Kelvin Lin
Track 1 

Image2World.py
Convert point in image-space (x, y) to world coordinates (GPS)
by applying the homography transform

"""

import numpy as np
import pickle
from DataIO import ld_camlink, sv_camlink
import sys
import cv2

# Global declarations
with open('./list_cam_u.txt') as f:
    FPList = f.read().splitlines()

fp = "./obj/text4.txt"

def main(argv):
    n_cam = 20
    n_layer = 2

    plist = ld_camlink(fp, n_cam, n_layer)
    if plist is None:
        exit(1)

    gps_list = list()
    for camid in range(4, n_cam):
        gps_list.append([])
        # Read the calibration file and convert the text format to a np.Matrix
        with open(FPList[camid] + 'calibration.txt', 'r') as f:
            mat_str = f.read().splitlines()

        # Calibration Matrix
        calib_coef = list(map(float, mat_str[0].replace(';', ' ').split(' ')))
        calib_mat = np.reshape(calib_coef, (3, 3))

        if len(mat_str) > 1:
            cap = cv2.VideoCapture(FPList[camid] + 'vdo.avi')
            cap_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Camera Frame Height
            cap_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # Camera Frame Width
            cap.release()

            dist_coef = list(map(float, mat_str[1].split(' ')))
            cam_coef = np.array([cap_W, 0, cap_W/2, 0, cap_W, cap_H/2, 0, 0, 1], dtype=np.float64)

            cam_Dmat = np.array(dist_coef, dtype=np.float32)
            cam_Kmat = np.reshape(cam_coef, (3, 3))

        # Retrieve points, flatten list to one list
        cam_pt = plist[camid]

        for side in cam_pt:
            # gps_list[camid].append([])

            # Convert point into GPS
            if len(mat_str) > 1:
                side = np.reshape([side], (1, -1, 2))
                side = side.astype(dtype=np.float32)
                ret = cv2.fisheye.undistortPoints(side, cam_Kmat, cam_Dmat)
                print()

            for pt in side:
                gps_pt = calib_mat

        print()



if __name__ == "__main__":
    main(sys.argv[1:])

