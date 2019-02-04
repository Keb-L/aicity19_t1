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
import Image2World
from functools import total_ordering
import ntpath

with open('./lib/list_cam_u.txt') as f:
    FPATH = f.read().splitlines()

# Globals
isactive = False
update = False
refPt = []

scale = 0.5

def main(argv):
    if argv:
        argv = list(map(int, argv))
    # annotate_image(argv, ldpath="./obj/cam_vect_1_20_temp.temp")
    # annotate_image(argv)
    # temp = DataIO.load_obj("./obj", "cam_vect.pkl")

    temp1 = ld_LabelVect("./data", "cam_vect_1_40_clean.txt")
    # temp2 = ld_LabelVect(*["./data", "cam_vect_1_20.tmp"])
    # sv_LabelVect("./data", "cam_vect_1_40_clean.txt", temp1)
    show_markedimg([1, 40], temp1, writeimg=True, iterate=True)
    # print()
    # temp2 = ld_LabelVect("./data", "cam_vect_21_33_upd.txt")

    # temp2.extend(temp1)
    # temp3 = LabeledVector.resetUID(temp2)

    # sv_LabelVect("./data", "cam_vect_21_40.txt", temp3)
    #
    # sv_LabelVect("./obj", "cam_vect2.txt", temp)
    # temp[5].UID = 5


    # temp = ld_LabelVect("./data/", "cam_vect_21_33_upd.txt", forceGPS=True)
    # show_markedimg(argv, temp, writeimg=True, iterate=True)
    # sv_LabelVect("./data/", "cam_vect_21_33_upd.txt", temp)
    print()
    # LabeledVector.resetUID(temp)

    # show_markedimg(argv, temp)

    # if not hasUniqueUID(temp):
    #     raise Exception("LabeledVect UID is not unique!")
    # print()



def annotate_image(camrange, svpath="./obj/cam_vect.txt", ldpath=None):
    """

    :param argv:
    :return:
    """
    en_list = ex_list = []
    sv_list = prevl = []

    if ldpath is not None:
        prevl = ld_LabelVect(*ntpath.split(ldpath))

    svloc, svfile = ntpath.split(svpath)
    svfile = os.path.splitext(svfile)[0]

    if not camrange:
        camrange = [1, 40]

    for camid in range(camrange[0] - 1, camrange[1]):
        vpath = FPATH[camid] + "vdo.avi"

        vinit = list(filter(lambda x: x.camid == camid+1, prevl))
        vinit_en, vinit_ex = LabeledVector.split_vlist(vinit)
        overlay = []
        # Type 0 = Enter, Type 1 = Exit
        ret_en, overlay = get_annotation(vpath, camid + 1, p_type=0, vinit=vinit_en)
        ret_ex, overlay = get_annotation(vpath, camid + 1, p_type=1, overlay=overlay, vinit=vinit_ex)

        # assert (len(ret_en) % 2 == 0 and len(ret_ex) % 2 == 0)

        # Update entry and exit lists
        while ret_en:
            vect = ret_en.pop()
            lvect = LabeledVector(vect, camid + 1, p_type=0)
            en_list.append(lvect)

        while ret_ex:
            vect = ret_ex.pop()
            lvect = LabeledVector(vect, camid + 1, p_type=1)
            ex_list.append(lvect)

        tmp = list()
        tmp.extend(en_list)
        tmp.extend(ex_list)
        sv_LabelVect(svloc, svfile + ".tmp", tmp, force=True)

    sv_list.extend(en_list)
    sv_list.extend(ex_list)

    sv_LabelVect(svloc, svfile + ".txt", sv_list)
    # DataIO.save_obj(master_list, "./obj/", "cam_vect.pkl")



def get_annotation(vpath, camid, p_type, overlay=None, vinit=None):
    """
    Image annotation script
    :param vpath:
    :param camid: true cam id
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

    if p_type == 0:
        color = (255, 0, 255)   # Magenta
        p_str = "entry"
    elif p_type == 1:
        color = (0, 255, 255)   # Yellow
        p_str = "exit"
    else:
        color = (255, 255, 0)   # Cyan
        raise Exception("Unknown point type!")


    cap = cv2.VideoCapture(vpath)
    ret, frame = cap.read()
    if not ret:
        print("An error has occurred in annotate_image!")
        return []

    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    winname = "Camera {0}: Select {1} point pairs...".format(camid, p_str)
    if overlay is None:
        overlay = np.zeros(frame.shape, np.uint8)

    cv2.imshow(winname, frame)
    cv2.setMouseCallback(winname, on_click, [winname, frame.copy(), overlay])

    for lvect in vinit:
        vect = [(int(x*scale), int(y*scale)) for (x, y) in lvect.vector]
        arrlist.append(vect)
        cv2.arrowedLine(overlay, *vect, color, thickness=2, line_type=4, tipLength=0.10)

    update = True

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
                cv2.arrowedLine(overlay, srtPt, endPt, color, thickness=2, line_type=4, tipLength=0.10)

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


def show_markedimg(argv, vlist, writeimg=False, iterate=False):
    """
    Overlays all labeled vectors on the first frame of each camera's video
    writeimg - dump labeled images to file
    iterate - automatically go through all images
    """
    if argv:
        argv = list(map(int, argv))
    else:
        argv = [1, 40]

    en_list, ex_list = LabeledVector.split_vlist(vlist)
    cam_min = argv[0] - 1
    cam_max = argv[1] - 1

    camid = cam_min
    while(True):
        vpath = FPATH[camid] + "vdo.avi"

        cap = cv2.VideoCapture(vpath)
        ret, frame = cap.read()   

        en_cam_list = list(filter(lambda x: x.camid == camid+1, en_list))
        ex_cam_list = list(filter(lambda x: x.camid == camid+1, ex_list))

        for v in en_cam_list:
            frame = cv2.arrowedLine(frame, v.vector[0], v.vector[1], (255, 0, 255), 
                                    thickness=4, line_type=4, tipLength=0.10)
            frame = cv2.putText(frame, "{0}-{1}".format(v.p_type, v.UID), org=v.center, 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2, color=(255, 0, 255))
        for v in ex_cam_list:
            frame = cv2.arrowedLine(frame, v.vector[0], v.vector[1], (0, 255, 255), 
                                    thickness=4, line_type=4, tipLength=0.10)
            frame = cv2.putText(frame, "{0}-{1}".format(v.p_type, v.UID), org=v.center, 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2, color=(0, 255, 255))

        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        winname = "Camera {0}: Annotated point pairs".format(camid+1)                            
        cv2.imshow(winname, frame)
        
        if writeimg:
            cv2.imwrite("./dump/c{0:03d}_labeled.jpg".format(camid+1), frame)

        while not iterate:
            kp = cv2.waitKey(1)
            if kp == ord('b'): # Arrow Left
                if camid == cam_min:
                    camid = cam_max
                else:
                    camid -= 1
                break
            if kp == ord('n'): # Arrow Right
                if camid == cam_max:
                    camid = cam_min
                else:
                    camid += 1
                break
            if kp == 27 or kp == 113 or kp == 81:  # ESC, q, Q:
                camid += 1
                break  

        cv2.destroyWindow(winname)

        if iterate:
            camid += 1
            if camid > cam_max:
                break
        


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


def sv_LabelVect(path, name, vlist, force=False):
    """
    Saves vector list to file system.
    Applies a unique operation on the vlist based on camid, p_type and vector
    :param path:
    :param name:
    :param vlist:
    :param force:
    :return:
    """
    vlist = uniqueVectors(vlist)
    fp = os.path.join(path, name)
    if os.path.isfile(fp):
        if not force and not DataIO.confirmOverride(fp):
            print("Aborted!")
            return None
    with open(fp, 'w') as f:
        for lvect in vlist:
            f.write(lvect.__str__() + "\n")
    if not force:
        print("Saved to " + fp)


def ld_LabelVect(path, name, forceGPS=False, vlist=None):
    """
    Read saved LabeledVectors and return a list of LabeledVectors
    """
    fp = os.path.join(path, name)
    if not os.path.isfile(fp):
        return None

    with open(fp, 'r') as f:
        rd = f.read().splitlines()

    ret = []
    for line in rd:
        data = [float(x) for x in line.split(' ')]
        data[0:7] = [int(x) for x in data[0:7]]
        vect = [(data[3], data[4]), (data[5], data[6])]
        
        # Without forced GPS
        if not forceGPS:
            ret.append(LabeledVector(vector=vect, camid=data[1], p_type=data[2], UID=data[0]))
        else:
        # # Forced GPS
            ret.append(LabeledVector(vector=vect, camid=data[1], p_type=data[2], UID=data[0], gps=data[7:9]))
    ret = uniqueVectors(ret)
    vlist = ret
    return vlist


def hasUniqueUID(vlist):
    en_list, ex_list = LabeledVector.split_vlist(vlist)
    # Compute the element difference
    ldiff = lambda l: len(l) - len(set([v.UID for v in l]))
    ret = ldiff(en_list) or ldiff(ex_list)  # True if there are copies
    return not ret


def uniqueVectors(vlist):
    vtuple = [(v.camid, v.p_type, v.vector[0], v.vector[1]) for v in vlist]
    vunique = list(set(vtuple))
    uindex = [vtuple.index(a) for a in vunique]

    if len(uindex) < len(vlist):
        print("WARNING: Unique operation has found duplicate vectors.\n\t{0} unique of {1} vectors".format(len(uindex), len(vlist)))
    uindex.sort()
    ret = [vlist[a] for a in uindex]
    return ret

@total_ordering
class LabeledVector:
    """
    Vector wrapper
    """
    _UIN = [0, 0]

    def __init__(self, vector, camid, p_type, UID=None, gps=None):
        """
        camid is the true camera id number
        """
        self.vector = vector
        self.center = (int((vector[0][0] + vector[1][0])/2), int((vector[0][1] + vector[1][1])/2))
        self.camid = camid
        self.p_type = p_type

        if gps is None:
            self.gps = Image2World.pt2world(camid-1, self.center, scale=scale)
        else:
            self.gps = gps

        if UID is None:
            self.UID = self._UIN[p_type]
            self._UIN[p_type] += 1
        else:
            self.UID = UID
            if UID > self._UIN[p_type]:
                self._UIN[p_type] = UID


    def __str__(self):
        """
        String override
        :return:
        """
        ret = "{0} {1} {2} ".format(self.UID, self.camid, self.p_type)
        ret += "{0:.0f} {1:.0f} {2:.0f} {3:.0f} ".format(self.vector[0][0], self.vector[0][1],
                                                    self.vector[1][0], self.vector[1][1])
        ret += "{0} {1} {2}".format(*self.gps)
        return ret

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
    def resetUID(cls, vlist, init=[0, 0]):
        """
        Given a list of LabeledVectors, orders them by camid and vector
        Re-assigns UID for both types starting at init. Class variable _UIN
        is updated to the next value
        """
        en_list, ex_list = LabeledVector.split_vlist(vlist)

        en_list.sort()
        ex_list.sort()

        for i in range(0, len(en_list)):
            en_list[i].UID = i + init[0]

        for i in range(0, len(ex_list)):
            ex_list[i].UID = i + init[1]

        cls._UIN = [init[0] + len(en_list), init[1] + len(ex_list)]
        ret = en_list
        ret.extend(ex_list)
        return ret
    
    @classmethod
    def split_vlist(cls, vlist):
        en_list = list(filter(lambda x: x.p_type == 0, vlist))
        ex_list = list(filter(lambda x: x.p_type == 1, vlist))
        return en_list, ex_list


if __name__ == "__main__":
    main(sys.argv[1:])
