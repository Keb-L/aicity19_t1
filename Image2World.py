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
import DataIO
import sys
import cv2


# Global declarations
with open('./lib/list_cam_u.txt') as f:
    FPATH = f.read().splitlines()
VRES = DataIO.ld_vidres('./lib/cam_res.txt')

fp = "./obj/track1_all.txt"

def main(argv):
    print()
    get_calibration(34)
    # n_cam = 40
    # n_layer = 1

    # plist = DataIO.ld_camlink(fp, n_cam, n_layer)
    # if plist is None:
    #     exit(1)

    # gps_list = list()
    # for camid in range(0, n_cam):
    #     gps_list.append([])

    #     [H, D, K] = get_calibration(camid)

    #     # Retrieve points, flatten list to one list
    #     cam_pt = plist[camid]

    #     for side in cam_pt:
    #         # gps_list[camid].append([])
    #         is_neg = [all(pt < 0) for pt in side]
    #         ret_pt = np.array([abs(pt) for pt in side])
    #         # Convert point into GPS
    #         if D is not None and K is not None:
    #             ret_pt = np.reshape([ret_pt], (1, -1, 2))
    #             ret_pt = cv2.fisheye.undistortPoints(ret_pt.astype(dtype=np.float64), K, D,
    #                                 None, None, K)
    #             print()

    #         for pt in side:
    #             gps_pt = np.linalg.inv(H) @ np.transpose(np.array([np.append(pt, 1)]))
    #             gps_pt = gps_pt / gps_pt[2]
    #             print()

    #     print()

def vlist2world(camid, vlist, scale=1):
    [H, D, K] = get_calibration(camid)

    for i in range(0, len(vlist)):
        gps_pt = pt2world(camid, vlist[i].center, scale)
        # vmidpt = np.array([[vlist[i].center]], dtype=np.float64)
        # if D is not None and K is not None:
        #     vmidpt = cv2.fisheye.undistortPoints(vmidpt, K, D, None, None, K)
        # gps_pt = np.linalg.inv(H) @ np.transpose([np.append(vmidpt, 1)])
        vlist[i].gps = gps_pt

    return vlist

def pt2world(camid, pt, scale=1):
    [H, D, K] = get_calibration(camid)

    pt = np.array([[pt]], dtype=np.float64) / scale
    if D is not None and K is not None:
        pt = cv2.fisheye.undistortPoints(pt, K, D, None, None, K)
    gps_pt = np.linalg.inv(H) @ np.transpose([np.append(pt, 1)])
    gps_pt = gps_pt / gps_pt[2]
    gps_pt = [x[0] for x in gps_pt]
    return gps_pt

def get_calibration(camid):
    VRES = DataIO.ld_vidres('./lib/cam_res.txt')

    # Read the calibration file and convert the text format to a np.Matrix
    with open(FPATH[camid] + 'calibration.txt', 'r') as f:
        mat_str = f.read().splitlines()

    # Calibration Matrix
    calib_coef = list(map(float, mat_str[0].replace(';', ' ').split(' ')))
    H = np.reshape(calib_coef, (3, 3))
    D = None
    K = None

    if len(mat_str) > 1:
        cap_H = VRES[camid][0]
        cap_W = VRES[camid][1]

        dist_coef = list(map(float, mat_str[1].split(' ')))
        cam_coef = np.array([cap_W, 0, cap_W / 2, 0, cap_W, cap_H / 2, 0, 0, 1], dtype=np.float64)

        D = np.array(dist_coef, dtype=np.float64)
        K = np.reshape(cam_coef, (3, 3))
    return [H, D, K]

if __name__ == "__main__":
    main(sys.argv[1:])

