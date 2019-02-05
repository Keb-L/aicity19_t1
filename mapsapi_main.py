import numpy as np
import cv2
import sys
import googlemaps
from datetime import datetime
import vidPOI as vp
from color_dict import html_color_codes as htmlcc
from collections import deque

colormap = {'black':    '0x000000',
            'gray':     '0x808080',
            'red':      '0xFF0000',
            'orange':   '0xFFA500',
            'yellow':   '0xFFFF00',
            'green':    '0x00FF00',
            'blue':     '0x0000FF',
            'cyan':     '0x00FFFF',
            'purple':   '0x800080',
            'magenta':  '0xFF00FF',
            'white':    '0xFFFFFF'}

def main(argv):
    gmaps = googlemaps.Client(key='AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY')




    vlist = vp.ld_LabelVect("./data", "cam_vect_1_40_Feb3.txt")
    scenelist = [list(range(1, 6)), list(range(6, 10)), list(range(10, 41))]


    vlist_s3 = list(filter(lambda x: x.camid in scenelist[2], vlist))

    colororder = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'cyan', 'magenta']

    marker_list = list()
    for camid in range(10, 41):
        vlist_cam = list(filter(lambda x: x.camid == camid, vlist_s3))
        marker_c = colormap[colororder[camid % len(colororder)]]

        cam_en, cam_ex = vp.LabeledVector.split_vlist(vlist_cam)

        en_ct = ex_ct = 0
        for lvect in cam_en:
            marker_lbl = chr(ord('A') + en_ct)
            marker_c = htmlcc[colororder[camid % len(colororder)]]

            marker_list.append([*lvect.gps[0:2], marker_lbl, marker_c])
            en_ct += 1

            print()

        for lvect in cam_ex:
            marker_lbl = chr(ord('0') + ex_ct)
            ex_ct += 1
            if ex_ct > 9:
                print("WARNING: Overflow on exit vector count (greater than 10)")
                ex_ct = 0
            marker_list.append([*lvect.gps[0:2], marker_lbl, marker_c])
            print()
    genMarkerQuery(marker_list)
    print()


def queryStaticAPI(key, size, scale, zoom, center):
    baseURL = 'https://maps.googleapis.com/maps/api/staticmap?'

    gmap_key = 'key={0}'.format(key)

    gmap_size = 'size={0}x{1}'.format(*size)
    gmap_scale = 'scale={0}'.format(scale)

    if isinstance(center, str):
        gmap_center = 'center={0}'.format(center)
    elif len(center) == 2:
        gmap_center = 'center={0},{1}'.format(*center)
    else:
        print('WARNING: center is not specified')
    gmap_zoom = 'zoom={0}'.format(zoom)


def genMarkerQuery(mlist):
    mlist.sort(key=lambda x: x[2] + x[3])
    mlist_queue = deque(mlist)

    mquery = list()
    while mlist_queue:
        marker = mlist_queue.popleft()
        marker_orig = marker[2] + marker[3]

        marker_str = "marker="
        # Marker colors
        marker_color = "color:{0}".format(marker[3])
        marker_lbl = "label:{0}".format(marker[2])
        marker_loc = "{0:.06f},{1:.06f}".format(*marker[0:2])

        marker_str += "%7C".join([marker_color, marker_lbl, marker_loc])
        if not mlist_queue:
            mquery.append(marker_str)
            break
        # Get the label-color
        marker_next = mlist_queue[0][2] + mlist_queue[0][3]
        while marker_next == marker_orig:
            if not mlist_queue:
                break
            marker = mlist_queue.popleft()
            marker_loc = "{0:.06f},{1:.06f}".format(*marker[0:2])
            marker_str += "%7C" + marker_loc
            marker_next = mlist_queue[0][2] + mlist_queue[0][3]
        mquery.append(marker_str)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])