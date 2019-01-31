"""

AICity 2019
Collection of Data I/O scripts
Kelvin Lin

DataIO.py

"""
import numpy as np

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
    lines = list()
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    if(n_cam * n_layer != len(lines)):
        print("Size mismatch between ")
        return None
    
    # Format line list into data structure 
    ret = list()
    camid = 0
    layerid = 0
    for i in range(0, len(lines)):
        l = lines[i]                # Get the line

        if layerid == n_layer:      # End of data for camera, move to next
            layerid = 0
            camid += 1

        if layerid == 0:            # Add a new "camera"
            ret.append([])
        ret[camid].append([])   # Add a new layer

        # Process line
        arr = np.fromstring(l, dtype=int, sep=' ') # Convert to array
        arr = np.reshape(arr, (-1, 2)) # Convert to list of arrays

        temp = list()
        for elem in arr:
            temp.append(elem)
        temp.pop()

        assert(np.array_equal(arr[-1], np.array([1, -1])))

        ret[camid][layerid] = temp
        layerid += 1

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

