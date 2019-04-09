"""

AICity 2019
Collection of Data I/O scripts
Kelvin Lin

DataIO.py

"""
import numpy as np
import sys
import cv2
import os, pickle

with open('./lib/list_cam_u.txt') as f:
    fpath = f.read().splitlines()

def main(argv):
    print()
    # sv_vidres('./lib/cam_res.txt')
    # ld_vidres('./lib/cam_res.txt')


def sv_vidres(filename):
    vres = list()
    for vpath in fpath:
        vpath = vpath + 'vdo.avi'

        cap = cv2.VideoCapture(vpath)

        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Camera Frame Height
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Camera Frame Width

        vres_str = "{:.0f} {:.0f}".format(height, width)
        vres.append(vres_str)

    with open(filename, 'w') as f:
        f.writelines('\n'.join(vres))
    print("Write complete")


def ld_vidres(filename, st=None):
    # Read file
    with open(filename, 'r') as f:
        vres = [int(x) for x in f.read().split()]
    # Reshape into (H, W) pairs
    vres = np.reshape(vres, (-1, 2))
    st = vres
    return vres


def sv_dictfile(filename, st):
    """
    Saves 2-layer dictionary st into filename

    st Structure
    > Camera
        > En/Ex
            > Points

    (Camera/En) Points (terminator)
    (Camera/Ex) Points (terminator)

    Data is saved in ASCII characters with a terminator string sequence at the end of each line
    """
    # Allowed types: dict
    if not isinstance(st, dict):
        return None

    # Retrieve list of keys
    kClist = st.keys()
    with open(filename, 'w') as f:
        for kC in kClist:    # Key Camera
            kDlist = st[kC].keys()
            string = ""
            for kD in kDlist:   # Key Direction ('en', 'ex')
                ptList = st[kC][kD]
                for pt in ptList:   # for each point
                    for x in pt:
                        if kD == "en":
                            x = x
                        else:
                            x = -x
                        string += str(int(x)) + " "
            string += "1 -1\n"
            print(string)
            f.write(string)


def ld_camlink(filename, n_cam=40, n_layer=4, st=None):
    """
    Camera Entry/Exit point load function
    :param filename:
    :param n_cam:
    :param n_layer:
    :param st: save pointer
    :return:
    """
    # lines = list()
    data = list()
    with open(filename, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.split(' ')])
        # lines = f.read().splitlines()

    # if(n_cam * n_layer != len(lines)):
    #     print("Size mismatch between ")
    #     return None

    # Format line list into data structure
    ret = dict()
    for line in data:
        line = [*[int(x) for x in line[:7]], *[x for x in line[7:]]]

        camid = line[1]
        vtype = line[2]

        ptlist = np.array([line[3:5], line[5:7]])
        if camid not in ret:
            ret[camid] = list()
        # if vtype not in ret[camid]:
        #     ret[camid][vtype] = list()
        if vtype:
            ptlist = -ptlist

        ret[camid].append(ptlist)

    # camid = 0
    # layerid = 0
    # for i in range(0, len(lines)):
    #     l = lines[i]                # Get the line
    #
    #     if layerid == n_layer:      # End of data for camera, move to next
    #         layerid = 0
    #         camid += 1
    #
    #     if layerid == 0:            # Add a new "camera"
    #         ret.append([])
    #     ret[camid].append([])   # Add a new layer
    #
    #     # Process line
    #     arr = np.fromstring(l, dtype=int, sep=' ') # Convert to array
    #     arr = np.reshape(arr, (-1, 2)) # Convert to list of arrays
    #
    #     temp = list()
    #     for elem in arr:
    #         temp.append(elem)
    #     temp.pop()
    #
    #     assert(np.array_equal(arr[-1], np.array([1, -1])))
    #
    #     ret[camid][layerid] = temp
    #     layerid += 1

    st = ret
    return st


def sv_camlink(filename, st):
    """

    :param filename: save file path
    :param st: list structure
    :return:
    """
    with open(filename, 'w') as f:
       n_cam = len(st)
       for cam in st:
           n_layer = len(cam)
           for p in cam:
                pl = list(map(list, p))  # Convert ndarray to list
                pl.append([1, -1])       # Add terminator
                pl_flat = [x for y in pl for x in y]    # Flatten
                wdata = " ".join(str(x) for x in pl_flat) + "\n"    # to space-delimited string
                f.write(wdata)


# Save an object to the file system
def save_obj(obj, path, name):
    fp = os.path.join(path, name)
    if os.path.isfile(fp):
        if not confirmOverride(fp):
            print("Aborted")
            return None
    if not os.path.exists(path):
        os.makedirs(path)
    with open(fp, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print('Saved file!')


# Load an object from the file system
def load_obj(path, name):
    fp = os.path.join(path, name)
    if os.path.isfile(fp):
        with open(os.path.join(path, name), 'rb') as f:
            print('Loaded file!')
            return pickle.load(f)
    return None

def confirmOverride(fp=""):
    kp = 0
    while(kp not in ('y', 'n')):
        kp = input("Confirm override of"+fp+"(y/n):")
    return kp == 'y'

if __name__ == "__main__":
    main(sys.argv[1:])