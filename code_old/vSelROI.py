'''

AI City 2019
Training Video Region-of-Interest Selection
Kelvin Lin (linkel1@uw.edu)

vSelROI.py

Command line calls
if not arguments are provided, then the script will go through every video
py -3 ./vSelROI.py
py -3 ./vSelROI.py [cam_id_start] [cam_id_end]

Manual annotation of the entry/exit bounding regions in the AIC Training Videos
The bounding regions are converted into list of the central points which
are defined to be the entry and exit points.

ROI are selected by drawing the bounding area and pressing SPACE or ENTER.
The process can be canceling by pressing c or ESC.

The data is implemented into a nested dictionary (stored in track1_road.pkl).
See Python pickle module for format and manipulation info.

The format of the nested dictionary is
dict (
    'CAMERA ID' : {
                    'en': [entrance points]
                    'ex': [exit points]
                    }
)

File System:
vSelROI.py
obj
> track1_road.pkl

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
with open('../list_cam.txt') as f:
    VIDPATH = f.read().splitlines()

DICTNAME = 'track1_road'
scale = 0.5

def main(argv):
    # Filter the video list using the command line arguments
    fVIDPATH = []
    if len(argv) == 2:
        cam_range = range(int(argv[0]), int(argv[1]) + 1)
        for i in cam_range:
            matches = [s for s in VIDPATH if 'c{:03d}'.format(i) in s]
            fVIDPATH.append('.' + matches[0])
    else:
        fVIDPATH = ['.' + s for s in VIDPATH]

    hmap = load_obj(DICTNAME)

    # Iterate through each video and perform the BB selection
    for vpath in fVIDPATH:
        vpath = vpath + VID_NAME

        # Get Camera ID
        cam_id = re.search("c([\d]{3})", vpath)[0]
        scn_id = re.search("S([\d]{2})", vpath)[0]

        # Instantiate Video Capture
        cap = cv2.VideoCapture(vpath)

        # Read the first frame
        ret, frame = cap.read()
        if not ret:     # if no frame read, continue
            continue
        # cv2.imshow(vpath, frame)    # show frame
        frame = cv2.putText(frame, scn_id + '/' + cam_id, org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2,
                    color=(255, 0, 255))

        frame = cv2.resize(frame, None, fx = scale, fy = scale, interpolation=cv2.INTER_CUBIC)

        # Selection of entry/exit points
        rects = cv2.selectROIs('{0}: select entrance locations'.format(cam_id), frame, fromCenter=True)
        cv2.destroyWindow('Select Entrance locations')
        en_pt = getPoints(rects)

        rects = cv2.selectROIs('{0}: select exit locations'.format(cam_id), frame, fromCenter=True)
        cv2.destroyWindow('Select Exit locations')
        ex_pt = getPoints(rects)

        # Free capture object, destroy all open windows
        cap.release()
        cv2.destroyAllWindows()

        # Save to dictionary
        # secondary dictionary for en(try) and ex(it) points
        subhmap = {'en': en_pt, 'ex': ex_pt}

        # Check to see if key is already in hash map
        # If exists, confirm override
        # else, add to map
        if cam_id in hmap.keys():
            if confirmOverride():   # Confirm override of key
                print('Confirmed override of existing entry!')
                hmap[cam_id] = subhmap
            else:
                print('Override aborted!')
        else:
            hmap[cam_id] = subhmap

        save_obj(hmap, DICTNAME)
        DataIO.sv_dictfile("obj/track1_en_ex.txt", hmap)
    #end for loop
    print('Done')
    print()


# Save an object to the file system
def save_obj(obj, name):
    if not os.path.exists('obj'):
        os.makedirs('obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print('Saved file!')


# Load an object from the file system
def load_obj(name):
    if os.path.isfile('obj/' + name + '.pkl'):
        with open('obj/' + name + '.pkl', 'rb') as f:
            print('Loaded file!')
            return pickle.load(f)
    else:
        return dict()


# Given a list of rectangles, reduce to the unique center points
def getPoints(rects):
    ret = list()
    for pt in rects:
        ret.append( ((pt[0] + pt[2] / 2)/scale, (pt[1] + pt[3] / 2)/scale) )
    ret = list(set(ret))
    return ret


def confirmOverride():
    kp = 0
    while(kp not in ('y', 'n')):
        kp = input('Confirm override of entry (y/n):')
    return kp == 'y'

if __name__ == "__main__":
    main(sys.argv[1:])