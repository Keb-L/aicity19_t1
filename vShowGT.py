import cv2 as cv2
import csv
import sys
from time import sleep


# Global declarations
VID_NAME = 'vdo.avi'
# VIDPATH = glob.glob('../train/S*/c*/' + VID_NAME)
with open('../list_cam.txt') as f:
    VIDPATH = f.read().splitlines()
scale = 0.5

def main(argv):
    if len(argv) != 1:
        print('Requires one command line argument for camera ID!')
        return(1)

    cam_id = int(argv[0])
    matches = [s for s in VIDPATH if 'c{:03d}'.format(cam_id) in s]
    campath = '.' + matches[0]

    gtpath = campath + 'gt/gt.txt'
    vpath = campath + 'vdo.avi'
    csv_file = open(gtpath, mode='r')

    # [frame, ID, left, top, width, height, 1, -1, -1, -1]
    gt = csv.reader(csv_file)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps
    # Bounding Region
    bb = list(map(int, gt.__next__()))

    fcount = 1
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Our operations on the frame come here
        cv2.putText(frame, str(fcount), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2,
                    color=(0, 255, 0))

        # Assume GT is in order
        while(int(bb[0]) <= fcount):
            id = bb[1]
            tlpt = (bb[2], bb[3])               # Top-left pt
            brpt = (bb[2]+bb[4], bb[3]+bb[5])   # Bottom-right pt

            cv2.putText(frame, str(id), org=tlpt,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2,
                    color=(255, 0, 255))

            cv2.rectangle(frame, pt1=tlpt, pt2=brpt, color=(255, 0, 255), thickness=2)
            bb = list(map(int, gt.__next__()))

        frame = cv2.resize(frame, None, fx = scale, fy = scale)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        kp = cv2.waitKey(1)
        if kp & 0xFF == ord('q'):
            break
        elif kp & 0xFF == ord('p'):
            while(True):
                kp = cv2.waitKey(1)
                if kp & 0xFF == ord('p'):
                    break

        fcount += 1
        sleep(delay)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])