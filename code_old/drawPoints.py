'''

AI City 2019
Training Video Interest Point 
Kelvin Lin (linkel1@uw.edu)

'''

import sys
import re
import os
import numpy as np
import cv2
import pickle
import DataIO

# Global declarations
VID_NAME = 'vdo.avi'
# VIDPATH = glob.glob('../train/S*/c*/' + VID_NAME)
with open('./lib/list_cam_test.txt') as f:
    VIDPATH = f.read().splitlines()

scale = 1
en_dump = True

def main(argv):
    if not argv:
        argv = [1, 40]
    cam_range = range(int(argv[0])-1, int(argv[1]))
    all_pt = DataIO.ld_camlink("./data/cam_vect_1_40_clean.txt", 40, 1)

    # for camid in cam_range:
    for vpath in VIDPATH:
        camid = int(*re.findall("c([\d]{3})", vpath)) - 1
        # vpath = VIDPATH[camid] + VID_NAME

         # Instantiate Video Capture
        cap = cv2.VideoCapture(vpath + VID_NAME)

        # Read the first frame
        ret, frame = cap.read()
        if not ret:     # if no frame read, continue
            continue

        ptlist = all_pt[camid+1][0]
        for pt in ptlist:
            if(all(pt < 0)): # Exit point
                frame = cv2.circle(frame, tuple(abs(pt)), 10, (255, 0, 0), -1)
            else:   # Entry point
                frame = cv2.circle(frame, tuple(pt), 10, (0, 0, 255), -1)

        frame = cv2.putText(frame, "Camera " + str(camid+1), org=(5, 40),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2,
                    color=(255, 0, 255))


        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(str(camid + 1), frame)
        if en_dump:
            if not os.path.exists("./dump"):
                os.mkdir("./dump")
            cv2.imwrite("./dump/"+"c{:03d}_label.jpg".format(camid+1), frame)

        kp = cv2.waitKey(0)
        if kp == ord('q'):
            break
        # Free capture object, destroy all open windows
        cap.release()
        cv2.destroyAllWindows()

    #end for loop
    print('Done')
    print()

if __name__ == "__main__":
    main(sys.argv[1:])