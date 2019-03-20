"""
AI City 2019
Track 1

Video labeling and compiled test video generation
"""
import cv2
import numpy as np
from color import Color

import random
import argparse
import uuid
import os

COLOR_SET = [c for c in dir(Color) if not c.__contains__('__')]
COLOR_NUM = 0

# Combined frame unit size
VHEIGHT = 900
VWIDTH = 1600

scale = 2

"""
Maintains the folders cfg, gt, labeled_vid, combined_vid
"""

def main(args):
    if args['action'] == 'label' or args['action'] == 'both':
        labelVid(args)
    if args['action'] == 'combine' or args['action'] == 'both':
        combine_video(args)


def labelVid(args):
    dirname = './labeled_vid/'
    mkdir_ifndef(dirname)

    for cam_id in args['camera']:

        bbpath = args['gt_path'] + 'c{:03d}.txt'.format(cam_id)

        id_color = dict()
        bb_ptr = 0
        # Read bounding boxes
        with open(bbpath, 'r') as f:
            bb_data = []
            for line in f:  # read rest of lines
                bb_data.append([int(x) for x in line[:-1].split(',')])

        # Enforce bounding box data is sorted in frame-order
        bb_data.sort(key=lambda x: x[0])

        # Setup Video path and VideoCapture object
        vpath = args['cam_path'][cam_id] + 'vdo.avi'
        cap = cv2.VideoCapture(vpath)

        frame_sz = (int(cap.get(3))//scale, int(cap.get(4))//scale)

        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        vout = cv2.VideoWriter(dirname + 'labeled_c{:03d}.avi'.format(cam_id), fourcc, 10.0, frame_sz)

        print("Processing camera {c}...\n"
              "Frame Size = {w}x{h}\t".format(c=cam_id, w=frame_sz[0], h=frame_sz[1]))

        # Read through each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Current frame number

            # Retrieve all bounding boxes that match
            bb_list = []
            for bb in bb_data[bb_ptr:]:
                if bb[0] != frame_num:
                    break
                bb = np.array(bb, dtype=np.int64) * np.array([1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0, 0])
                bb_list.append(bb.astype(np.int64))
            bb_ptr += len(bb_list)

            frame = cv2.resize(frame, None, fx=1/scale, fy=1/scale)

            # Apply frame annotations
            for bb in bb_list:
                color = get_color(id_color, bb[1])
                frame = draw_rect(frame, *bb[2:6], color)
                frame = cv2.putText(frame, "{id}".format(id=bb[1]), tuple(bb[2:4]), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.1,
                                    color=color, thickness=2)

            frame = cv2.putText(frame, "{f}".format(f=int(frame_num)), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1.5, color=Color.green(), thickness=2)
            # frame = cv2.putText(frame, "c{:03d}".format(int(cam_id)), (int(cap.get(3)//2 - 125), 50), cv2.FONT_HERSHEY_DUPLEX,
            #                     fontScale=1.5, color=Color.green(), thickness=2)
            #
            vout.write(frame)
            # cv2.imshow("Frame", frame)
            # kp = cv2.waitKey(0)
            # if kp == ord('n'):
            #     break
            # elif kp == ord('q'):
            #     exit(0)
        cap.release()
        vout.release()



def combine_video(args):
    counter = 1
    dirname = "./combined_vid/"
    mkdir_ifndef(dirname)
    for vset in args['vid_set']:
        nvid = len(vset)//2
        vid_id = np.array([int(x) for x in vset[1::2]])
        vid_pos = np.array(vset[::2])

        print("Processing Video Set {:d}...".format(counter))

        # Determine the number of blocks needed
        width_block = max(vid_pos % 10) + 1
        height_block = int(max(vid_pos // 10)) + 1
        frame_sz = (int(VWIDTH * width_block), int(VHEIGHT * height_block))

        fourcc = cv2.VideoWriter_fourcc(*'MP42')

        vout = cv2.VideoWriter(dirname + str(uuid.uuid4()) + '.avi', fourcc, 10.0, frame_sz)

        # Create VideoCapture objects
        vc_list = []
        for cam_id in vid_id:
            vc_list.append(cv2.VideoCapture('./labeled_vid/labeled_c{:03d}.avi'.format(cam_id)))

        vc_done = np.zeros(len(vc_list))
        while True:
            # All Video Captures are out of frames
            if np.all(vc_done):
                break

            framelist = []
            for i in range(0, nvid):
                cap = vc_list[i]
                ret, frame = cap.read()
                if not ret:
                    vc_done[i] = 1  # Mark this VideoCapture as done
                    frame = np.zeros((int(cap.get(3)), int(cap.get(4)), 3), dtype=np.uint8)

                frame = cv2.putText(frame, "c{:03d}".format(vid_id[i]), (int(cap.get(3))-150, 50), cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=1.5, color=Color.green(), thickness=2)

                frame = resize_frame(frame, VHEIGHT)
                # frame = resize_frame(frame, VWIDTH)
                framelist.append(frame)
                # cv2.imshow("c{:03d}".format(vid_id[i]), frame)
            # cv2.waitKey(0)
            #
            comp_frame = vid_concat(framelist, vid_pos)  # Composite Frame
            vout.write(comp_frame)
            # cv2.imshow("cat", comp_frame)
            # kp = cv2.waitKey()
            # if kp == ord('q'):
            #     exit(0)
            # elif kp == ord('n'):
            #     break

        # Complete, release all VideoCapture and VideoWriter
        for vc in vc_list:
            vc.release()
        vout.release()
        counter += 1

def mkdir_ifndef(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def resize_frame(frame, targ):
    fh, fw, _ = frame.shape
    frame_new = frame.copy()

    ratio = targ / fh
    dW = int(fw * ratio)
    dH = int(fh * ratio)
    return cv2.resize(frame_new, dsize=(dW, dH))


def vid_concat(framelist, vid_pos):
    w_blocks = int(max(vid_pos % 10)) + 1
    h_blocks = int(max(vid_pos // 10)) + 1

    comp_frame = np.zeros((h_blocks*VHEIGHT, w_blocks*VWIDTH, 3), dtype=np.uint8)

    for i in range(0, len(framelist)):
        row_pos = vid_pos[i] // 10
        col_pos = vid_pos[i] % 10

        frame = framelist[i]
        frame_h, frame_w, _ = frame.shape


        row_st = int(row_pos*VHEIGHT + (VHEIGHT - frame_h) // 2)
        row_ed = int(row_pos*VHEIGHT + frame_h - (VHEIGHT - frame_h) // 2)

        col_st = int(col_pos*VWIDTH + (VWIDTH - frame_w)//2)
        col_ed = int(col_pos*VWIDTH + frame_w + (VWIDTH - frame_w)//2)

        comp_frame[row_st:row_ed, col_st:col_ed] = frame
    return comp_frame

def get_color(id_color, query):
    global COLOR_NUM
    if query in id_color:
        color = id_color[query]
    else:
        # color = getattr(Color, COLOR_SET[random.randint(0, len(COLOR_SET)-1)])()
        color = getattr(Color, COLOR_SET[COLOR_NUM % len(COLOR_SET)])()
        id_color[query] = color
        COLOR_NUM += 1
    return color


def draw_rect(image, x, y, h, w, color):
    return cv2.rectangle(image, (x, y), (x+h, y+w), color, thickness=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--camera", action="store", default=None, help="Camera ID (text file)")
    # parser.add_argument("-t", "--type", action="store", default="test", help="Camera mode (train/test)")
    parser.add_argument("-d", "--delimiter", action="store", default=" ", help="file delimiter (default space)")
    parser.add_argument("-cf", "--camera_file", action="store", default="./cfg/list_cam_test.txt", help="camera path file")
    parser.add_argument("-do", action="store", default="both", help="actions to do (label, combine, both)")

    parser.add_argument("-gt", "--ground_truth", action="store", default="./gt/", help="Ground truth bounding box path")        # Labeled Videos config
    parser.add_argument("-cfg", "--config_file", action="store", default="./cfg/vset.txt", help="compiled video config file")   # Combine Videos config

    parser.add_argument("-scale", "--scale_factor", action="store", default=0.5, help="video resolution scale factor")



    argv = parser.parse_args()


    args = dict()

    # args['type'] = argv.type

    args['camera'] = []
    if argv.camera is None:
        args['camera'] = [*range(6, 11), 10, *range(16, 30), *range(33, 37)]
    else:
        with open(argv.camera, 'r') as f:
            for line in f:
                args['camera'] += [int(x) for x in line.split(argv.delimiter)]

    args['gt_path'] = argv.ground_truth
    args['action'] = argv.do


    with open(argv.config_file, 'r') as f:
        args['vid_set'] = []
        for line in f:
            args['vid_set'].append([float(x) for x in line.split(argv.delimiter)])


    with open(argv.camera_file, 'r') as f:
        args['cam_path'] = f.read().splitlines()

    random.shuffle(COLOR_SET)
    main(args)
