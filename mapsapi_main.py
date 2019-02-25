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
    if argv:
        argv = list(map(int, argv))
    gmaps = googlemaps.Client(key='AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY')

    with open("./data/waypoints_1_40.txt") as f:
        gpslist = f.read().splitlines()
    gpslist = [np.fromstring(x, sep=' ', dtype=np.float64) for x in gpslist]

    # Camera filter
    # np.arange(start, end+1)
    # gpslist = list(filter(lambda x: x[1] in np.arange(21, 40+1), gpslist))

    # Waypoint filter, np.arange(start, end + 1)
    # Waypoint groups:
    #   0, 5
    #   5, 10
    #
    gpslist = list(filter(lambda x: x[0] in np.arange(0, 5+1), gpslist))

    print()
    # vlist = vp.ld_LabelVect("./data", "cam_vect_1_40_clean_edited.txt")

    # vlist_f = list(filter(lambda x: x.camid in np.arange(argv[0], argv[1]+1), vlist))

    # # colororder = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'cyan', 'magenta']
    # colororder = ['black', 'brown', 'green', 'purple', 'yellow', 'blue', 'gray', 'orange', 'red', 'white']

    # marker_list = list()
    # for camid in range(argv[0], argv[1]+1):
    #     vlist_cam = list(filter(lambda x: x.camid == camid, vlist_f))
    #     marker_c = colororder[camid % len(colororder)]

    #     cam_en, cam_ex = vp.LabeledVector.split_vlist(vlist_cam)

    #     en_ct = ex_ct = 0
    #     for lvect in cam_en:
    #         marker_lbl = chr(ord('A') + en_ct)
    #         # marker_c = htmlcc[colororder[camid % len(colororder)]]
    #         # marker_c = color_set[camid % len(color_set)]

    #         marker_list.append([*lvect.gps[0:2], marker_lbl, marker_c])
    #         en_ct += 1

    #     for lvect in cam_ex:
    #         marker_lbl = chr(ord('0') + ex_ct)
    #         ex_ct += 1
    #         if ex_ct > 9:
    #             print("WARNING: Overflow on exit vector count (greater than 10)")
    #             ex_ct = 0
    #         marker_list.append([*lvect.gps[0:2], marker_lbl, marker_c])


    apikey = 'AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY'
    mapsize = [640, 640]
    sz = None # {None, small, mid, tiny}
    mapscale = 2
    mapzoom = 16
    mapcenter = []#[42.498780, -90.686393]

    mquery = genSimpleMarkerQuery2(gpslist)
    # mquery = genMarkerQuery(marker_list, sz=None)
    queryStaticAPI(key=apikey, size=mapsize, scale=mapscale, zoom=mapzoom, center=mapcenter, markers=mquery)


def queryStaticAPI(key, size, scale, zoom, center, markers):
    baseURL = 'https://maps.googleapis.com/maps/api/staticmap?'

    gmap_key = 'key={0}'.format(key)

    gmap_size = 'size={0}x{1}'.format(*size)
    gmap_scale = 'scale={0}'.format(scale)

    if isinstance(center, str):
        gmap_center = 'center={0}'.format(center)
    elif len(center) == 2:
        gmap_center = 'center={0},{1}'.format(*center)
    else:
        gmap_center = ''
        print('WARNING: center is not specified')
    gmap_zoom = 'zoom={0}'.format(zoom)

    gmap_markers = "&".join(markers)
    # gmap_markers = markers[0]

    fullquery = baseURL + "&".join([gmap_size, gmap_scale, gmap_zoom, gmap_center, gmap_markers, gmap_key])

    if len(fullquery) > 2000:
        print("Note: query is {0} characters long!".format( len(fullquery) ))
    
    with open("queryurl.txt", "w") as f:
        f.write(fullquery)


def genSimpleMarkerQuery2(gpslist):
    gps_queue = deque(gpslist)
    i = 0

    colororder = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white']

    mquery = list()
    while gps_queue:
        waypoint = gps_queue.popleft()
        gps = waypoint[2:4]

        cam_color = int(waypoint[0]) // 10
        cam_lbl = int(waypoint[0]) % 10

        marker_str = "markers="
        marker_color = "color:{0}".format(colororder[cam_color])
        marker_lbl = "label:{0}".format(chr(ord('0') + cam_lbl))
        marker_loc = "{0:.06f},{1:.06f}".format(*gps)
        marker_size = "size:{0}".format('mid')

        marker_str += "%7C".join([marker_lbl, marker_color, marker_loc, marker_size])

        # if not gps_queue:
        mquery.append(marker_str)
            # break
    return mquery


def genSimpleMarkerQuery(gpslist):
    gps_queue = deque(gpslist)
    i = 0

    mquery = list()
    while gps_queue:
        gps = gps_queue.popleft()

        marker_str = "markers="
        marker_lbl = "label:{0}".format(chr(ord('A') + i))
        marker_loc = "{0:.06f},{1:.06f}".format(*gps[0:2])

        marker_str += "%7C".join([marker_lbl, marker_loc])

        if not gps_queue:
            mquery.append(marker_str)
            break
    return mquery

def genMarkerQuery(mlist, sz):
    """
    Generates marker query string
    """
    mlist.sort(key=lambda x: x[2] + x[3])
    mlist_queue = deque(mlist)

    mquery = list()
    while mlist_queue:
        marker = mlist_queue.popleft()
        marker_orig = marker[2] + marker[3]

        marker_str = "markers="
        # Marker colors
        # sz = 'small'
        # sz = None
        marker_size = "size:{0}".format(sz)
        marker_color = "color:{0}".format(marker[3])
        marker_lbl = "label:{0}".format(marker[2])
        marker_loc = "{0:.06f},{1:.06f}".format(*marker[0:2])

        if sz is None:
            marker_str += "%7C".join([marker_color, marker_lbl, marker_loc])
        elif sz != 'tiny':
            marker_str += "%7C".join([marker_size, marker_color, marker_lbl, marker_loc])
        else:
            marker_str += "%7C".join([marker_size, marker_color, marker_loc])

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
    # temp = "&".join(mquery)
    return mquery


if __name__ == "__main__":
    main(sys.argv[1:])