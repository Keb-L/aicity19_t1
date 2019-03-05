import numpy as np
import cv2
import sys
import googlemaps
from datetime import datetime
import vidPOI as vp
from color_dict import html_color_codes as htmlcc
from collections import deque
import json
import uuid
import urllib.request

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
<<<<<<< HEAD
    gmaps = googlemaps.Client(key='AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY')

    with open("./data/waypoints_1_40.txt") as f:
        gpslist = f.read().splitlines()
    gpslist = [np.fromstring(x, sep=' ', dtype=np.float64) for x in gpslist]
=======
    # gmaps = googlemaps.Client(key='AIzaSyBLWctOyJYmEg4j-sZIWxvBWswzgIFUd2U')

    with open("./data/waypoints_1_40_v3.txt", "r") as f:
        gpslist = f.read().splitlines()
    gpslist = [np.fromstring(x, sep=' ', dtype=np.float64) for x in gpslist]

    with open("./data/wp_gps_pairs.txt", "r") as f:
        wp_gps = f.read().splitlines()
    wp_gps = [np.fromstring(x, sep=' ', dtype=np.float64) for x in wp_gps]
    # wp_gps = [x.split(sep=" ") for x in wp_gps]

    with open("./data/wp_pairs.txt", "r") as f:
        wplist = f.read().splitlines()
    wplist = [np.fromstring(x, sep=' ', dtype=np.int) for x in wplist]


    # Read json
    with open("./data/gmaps_dir_out.json", "r") as f:
        path_dict = f.read()
    path_dict = json.loads(path_dict)

    # staticAPI_poly_main(path_dict)
    # Directions API calls
    # gmaps_dir_main(wplist, wp_gps)

    # Static API calls
    staticAPI_marker_main(gpslist)

def gmaps_dir_main(wplist, wp_gps):
    """
    Outputs the directions API queries to a json file

    :param wplist:
    :param wp_gps: waypoint pairs list
    :return:
    """
    gmaps = googlemaps.Client(key='AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY')

    # req = googlemaps.directions()
    # Define directions parameters
    mode = "driving"    # Driving
    units = "metric"    # Metric Units
    language = "en"     # English
    departure_time = "now"
    traffic_model = "best_guess"

    # origin = [",".join(x[0:2]) for x in wp_gps]
    # dest = [",".join(x[2:4]) for x in wp_gps]
    origin = [x[0:2] for x in wp_gps]
    dest = [x[2:4] for x in wp_gps]

    # query = gmaps.distance_matrix(origins=origin, destinations=dest,
    #                               mode=mode, units=units, language=language, departure_time=departure_time,
    #                               traffic_model=traffic_model)
    query_out = dict()
    i = 0
    for o, d in zip(origin, dest):
        query = gmaps.directions(origin=o, destination=d,
                                  mode=mode, units=units, language=language, departure_time=departure_time, traffic_model=traffic_model)

        query_out[" ".join(map(str, wplist[i]))] = query
        i += 1
        # fid.write(json.dumps(query))

    fid = open("./data/gmaps_dir_out.json", "w")
    fid.write(json.dumps(query_out))
    fid.close()
    print()
>>>>>>> d6ffb317e7ace769f732eccfa21be0e1ec796070

    # Camera filter
    # np.arange(start, end+1)
    # gpslist = list(filter(lambda x: x[1] in np.arange(21, 40+1), gpslist))

<<<<<<< HEAD
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

=======
def staticAPI_poly_main(path_dict):
    path_str = list()
    for k in path_dict:
        v = path_dict[k]
        path_str.append(v[0]['overview_polyline']['points'])


    baseURL = 'https://maps.googleapis.com/maps/api/staticmap?'

    apikey = 'AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY'
    mapsize = [640, 640]
    mapstyle = 'style=feature:all|element:labels|visibility:off'
    sz = None  # {None, small, mid, tiny}
    mapscale = 2
    mapzoom = 15
    mapcenter = [42.496553, -90.684858]

    gmap_key = 'key={0}'.format(apikey)

    gmap_size = 'size={0}x{1}'.format(*mapsize)
    gmap_scale = 'scale={0}'.format(mapscale)
    gmap_zoom = 'zoom={0}'.format(mapzoom)
    gmap_center = 'center={0},{1}'.format(*mapcenter)
    gmap_paths = ["path=color:{0}%7Cweight:{1}%7Cenc:{2}".format("0x0000ff80", 3, x) for x in path_str]

    fullquery = baseURL + "&".join([gmap_size, mapstyle, gmap_scale, gmap_zoom, gmap_center, *gmap_paths, gmap_key])
    retrieve_image(fullquery)
    # with open("queryurl.txt", "w") as f:
    #     f.write(fullquery)


def staticAPI_marker_main(gpslist):

    # Camera filter
    # np.arange(start, end+1)
    # gpslist = list(filter(lambda x: x[1] in np.arange(21, 40+1), gpslist))

    # Waypoint filter, np.arange(start, end + 1)
    # Waypoint groups:
    #   0, 5
    #   5, 10
    #


    # @Charles change this range!
    gpslist = list(filter(lambda x: (x[0] % 100)in np.arange(4, 7+1), gpslist))

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
>>>>>>> d6ffb317e7ace769f732eccfa21be0e1ec796070

    apikey = 'AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY'
    mapsize = [640, 640]
    sz = None # {None, small, mid, tiny}
    mapscale = 2
    mapzoom = 16
    mapcenter = []#[42.498780, -90.686393]
<<<<<<< HEAD
=======

    mquery = genSimpleMarkerQuery2(gpslist)
    # mquery = genMarkerQuery(marker_list, sz=None)
    queryStaticAPI(key=apikey, size=mapsize, scale=mapscale, zoom=mapzoom, center=mapcenter, markers=mquery)
>>>>>>> d6ffb317e7ace769f732eccfa21be0e1ec796070

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
<<<<<<< HEAD
    
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


=======
    retrieve_image(fullquery)
    # with open("queryurl.txt", "w") as f:
    #     f.write(fullquery)


def genSimpleMarkerQuery2(gpslist):
    gps_queue = deque(gpslist)
    i = 0

    colororder = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white']

    mquery = list()
    while gps_queue:
        waypoint = gps_queue.popleft()
        gps = waypoint[2:4]

        cam_color = (int(waypoint[0]) % 100) // 10
        cam_lbl = chr(ord('0') + int(waypoint[0]) % 10)
        # if cam_lbl == '9':
        #     cam_lbl = 'A'
        marker_str = "markers="
        marker_color = "color:{0}".format(colororder[cam_color])
        marker_lbl = "label:{0}".format(cam_lbl)
        marker_loc = "{0:.06f},{1:.06f}".format(*gps)
        marker_size = "size:{0}".format('tiny')

        marker_str += "%7C".join([marker_lbl, marker_color, marker_size, marker_loc])

        # if not gps_queue:
        mquery.append(marker_str)
            # break
    return mquery


>>>>>>> d6ffb317e7ace769f732eccfa21be0e1ec796070
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

def retrieve_image(query, svpath=None):
    if svpath is None:
        svpath = "temp/" + str(uuid.uuid4().hex) + ".png"

    print("Writing to {0}".format(svpath))
    ret = urllib.request.urlretrieve(query)
    cv2.imwrite(svpath, cv2.imread(ret[0]))


if __name__ == "__main__":
    main(sys.argv[1:])