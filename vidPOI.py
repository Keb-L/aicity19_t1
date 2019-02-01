"""

AI City 2019
Kelvin Lin

vidPOI.py
Video Point-of-Interest Selector

This program is designed to draw entry/exit vectors on the first frame of a given video.

Controls:
LMB - draw a point
RMB - Erase a point
Enter - Convert most recent point pair into a vector
Backspace - Erase a vector
ESC/q/Q - quit


"""

import numpy as np
import cv2
import sys
import DataIO
import os
from functools import total_ordering

with open('./lib/list_cam_u.txt') as f:
    FPATH = f.read().splitlines()

# Globals
isactive = False
update = False
refPt = []

scale = 0.5

def main(argv):
    # annotate_image(argv)
    # temp = DataIO.load_obj("./obj", "cam_vect.pkl")
    # sv_LabelVect("./obj", "cam_vect.txt", temp)
    temp = ld_LabelVect("./obj", "cam_vect.txt")
    temp[5].UID = 5

    LabeledVector.resetUIN(temp)

    if not hasUniqueUID(temp):
        raise Exception("LabeledVect UID is not unique!")
    print()



def annotate_image(argv):
    """

    :param argv:
    :return:
    """
    en_list = []
    ex_list = []

    en_count = 0
    ex_count = 0

    master_list = []

    if argv:
        argv = list(map(int, argv))

    for camid in range(argv[0] - 1, argv[1]):
        vpath = FPATH[camid] + "vdo.avi"

        overlay = []
        # Type 0 = Enter, Type 1 = Exit
        ret_en, overlay = get_annotation(vpath, camid + 1, type=0)
        ret_ex, overlay = get_annotation(vpath, camid + 1, type=1, overlay=overlay)

        assert (len(ret_en) % 2 == 0 and len(ret_ex) % 2 == 0)

        # Update entry and exit lists
        for e in ret_en:
            vect = ret_en.pop()
            lvect = LabeledVector(vect, camid + 1, "entry")
            en_list.append(lvect)
            en_count += 1

        for e in ret_ex:
            vect = ret_ex.pop()
            lvect = LabeledVector(vect, camid + 1, "exit")
            ex_list.append(lvect)
            ex_count += 1

    master_list.extend(en_list)
    master_list.extend(ex_list)

    DataIO.save_obj(master_list, "./obj/", "cam_vect.pkl")



def get_annotation(vpath, camid, type, overlay=None):
    """
    Image annotation script
    :param vpath:
    :param camid:
    :param type:
    :param overlay:
    :return:
    """
    global isactive, refPt, update, scale
    isactive = False    # Reset to initial
    refPt = []          # Reset to initial
    update = True

    # Return structure
    arrlist = []

    if type == "entry":
        color = (255, 0, 255)   # Magenta
    elif type == "exit":
        color = (0, 255, 255)   # Cyan
    else:
        color = (255, 255, 0)   # Yellow


    cap = cv2.VideoCapture(vpath)
    ret, frame = cap.read()
    if not ret:
        print("An error has occurred in annotate_image!")
        return []

    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    winname = "Camera {0}: Select {1} point pairs...".format(camid, type)
    if overlay is None:
        overlay = np.zeros(frame.shape, np.uint8)
    cv2.imshow(winname, frame)
    cv2.setMouseCallback(winname, on_click, [winname, frame.copy(), overlay])

    while (True):
        kp = cv2.waitKey(1)
        if kp == 13:  # Enter key pressed
            # Verify length
            if refPt and len(refPt) % 2 == 0:  # is even
                endPt = refPt.pop()
                srtPt = refPt.pop()

                # Erase the points
                cv2.circle(overlay, srtPt, 5, (0, 0, 0), -1)
                cv2.circle(overlay, endPt, 5, (0, 0, 0), -1)

                # Draw an arrow
                cv2.arrowedLine(overlay, srtPt, endPt, color, thickness=2, line_type=4, tipLength=0.05)

                arrlist.append([srtPt, endPt])
                update = True
        elif kp == 8:   # Backspace
            if arrlist:
                vect = arrlist.pop()    # Remove last drawn vector
                endPt = vect.pop()      # Retrieve end point
                srtPt = vect.pop()      # Retrieve start point
                cv2.arrowedLine(overlay, srtPt, endPt, (0, 0, 0), thickness=2, line_type=4, tipLength=0.10)
                update = True
        elif kp == 27 or kp == 113 or kp == 81:  # ESC, q, Q:
            break

    cap.release()
    cv2.destroyWindow(winname)

    for i in range(0, len(arrlist)):
        arrlist[i] = [tuple(c/scale for c in pt) for pt in arrlist[i]]
    return arrlist, overlay


def on_click(event, x, y, flags, param):
    """
    Handles imshow window behavior
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global refPt, isactive, update
    [winname, frame, overlay] = param

    if event == cv2.EVENT_LBUTTONDOWN:    # LMB pressed
        refPt.append((x, y))
        if not isactive:
            cv2.circle(overlay, refPt[-1], 5, (255, 0, 0), -1)
            isactive = True
            update = True
        else:
            cv2.circle(overlay, refPt[-1], 5, (0, 0, 255), -1)
            isactive = False
            update = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if refPt:
            rmPt = refPt.pop()
            cv2.circle(overlay, rmPt, 5, (0, 0, 0), -1)
            isactive = not isactive
            update = True

    if update:
        ret, mask = cv2.threshold(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
        frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        cv2.imshow(winname, cv2.add(frame, overlay))
        update = False


def sv_LabelVect(path, name, vlist):
    fp = os.path.join(path, name)
    if os.path.isfile(fp):
        if not DataIO.confirmOverride():
            print("Aborted!")
            return None
    with open(fp, 'w') as f:
        for lvect in vlist:
            f.write(lvect.__str__() + "\n")
    print("Saved!")


def ld_LabelVect(path, name, vlist=None):
    fp = os.path.join(path, name)
    if not os.path.isfile(fp):
        return None

    with open(fp, 'r') as f:
        rd = f.read().splitlines()

    ret = []
    for line in rd:
        data = [int(x) for x in line.split(' ')]
        vect = [(data[3], data[4]), (data[5], data[6])]
        ret.append(LabeledVector(vector=vect, camid=data[1], type=data[2], UID=data[0]))
    vlist = ret
    return vlist


def hasUniqueUID(vlist):
    en_list = list(filter(lambda x: x.type == 0, vlist))
    ex_list = list(filter(lambda x: x.type == 1, vlist))
    # Compute the element difference
    ldiff = lambda l: len(l) - len(set([v.UID for v in l]))
    ret = ldiff(en_list) or ldiff(ex_list)  # True if there are copies
    return not ret

@total_ordering
class LabeledVector:
    """
    Vector wrapper
    """
    _UIN = [0, 0]

    def __init__(self, vector, camid, type, UID=None):
        self.vector = vector
        self.camid = camid
        self.type = type
        if UID is None:
            self.UID = self._UIN[type]
            self._UIN[type] += 1
        else:
            self.UID = UID
            if UID > self._UIN[type]:
                self._UIN[type] = UID


    def __str__(self):
        """
        String override
        :return:
        """
        ret = "{0} {1} {2} ".format(self.UID, self.camid, self.type)
        ret += "{0:.0f} {1:.0f} {2:.0f} {3:.0f}".format(self.vector[0][0], self.vector[0][1],
                                                    self.vector[1][0], self.vector[1][1])
        return

    def __cmp__(self, other):
        """
        Comparator override
        :param other:
        :return:
        """
        return (self.camid > other.camid) - (self.camid < other.camid)

    def __eq__(self, other):
        return (self.camid == other.camid) and (self.vector == other.vector)

    def __lt__(self, other):
        return (self.camid < other.camid) or ((self.camid == other.camid) and (self.vector < other.vector))


    @classmethod
    def init_UIN(cls, init):
        cls._UIN = init

    @classmethod
    def resetUIN(cls, vlist, init=[0, 0]):
        """
        Given a list of LabeledVectors, orders them by camid and vector
        Re-assigns UID for both types starting at init. Class variable _UIN
        is updated to the next value
        """
        en_list = list(filter(lambda x: x.type == 0, vlist))
        ex_list = list(filter(lambda x: x.type == 1, vlist))

        en_list.sort()
        ex_list.sort()

        for i in range(0, len(en_list)):
            en_list[i].UID = i + init[0]

        for i in range(0, len(ex_list)):
            ex_list[i].UID = i + init[1]

        cls._UIN = [init[0] + len(en_list), init[1] + len(ex_list)]
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
