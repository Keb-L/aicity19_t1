"""
AI City 2019
Track 1

Kelvin Lin
Charles Fang

"""
import numpy as np
import cv2
import sys
import googlemaps

from Node import Node

import collections
import argparse
import uuid
import hashlib
import urllib
import json
import os
import datetime, time
import matplotlib.pyplot as plt
import csv

gbl_gmap_zoom = 17  # Google Maps Zoom factor (global)

def main(argv):
    """
    --setup: run configuration setup
    --s/--summary: generate summary file
    --m/--marker: generate marker plot
    --p/--polyline: generate polyline plot

    -s <id start> <id end>: Scenario ID
    -c <id start> <id end>: camera range
    -z <zoom level>: Google Maps zoom level (0-20) Normally 15-17

    :param argv:
    :return:
    """
    global gbl_gmap_zoom
    # Creates configuration file, exits on completion.

    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", default=False, help="run configuration setup")
    ap.add_argument("-r", "--summary", action="store_true", default=False, help="generate summary file")
    ap.add_argument("-cp", "--cam_pair", action="store_true", default=True, help="generate camera pair data file")
    ap.add_argument("-m", "--marker", action="store_true", default=False, help="generate marker plot")
    ap.add_argument("-p", "--polyline", action="store_true", default=False, help="generate polyline plot")

    ap.add_argument("-s", "--scenario", nargs=2, default=[1, 5], help="Scenario ID")
    ap.add_argument("-c", "--camera", nargs=2, default=[1, 40], help="Camera range")
    ap.add_argument("-z", "--zoom", nargs=1, default=17, help="Google Maps zoom level")
    arg_in = ap.parse_args()

    if arg_in.setup:
        config_setup()

    """
    cfg structure
        camera: camera-to-waypoint mapping
        directions: Directions result for each pair
        pair: waypoint pairs
        gps: waypoint gps
        scenario: scenario-to-camera
        hash: hash string -> Determines auto-generated filename suffix
    """
    cfg = config_init()

    if arg_in.scenario:
        s_id = range(*map(int, arg_in.scenario))

    if arg_in.camera:
        c_id = range(*map(int, arg_in.camera))

    if arg_in.zoom:
        gbl_gmap_zoom = int(arg_in.zoom)
    """Static polyline/route plot method
    s_id: Scenario ID
    cam_range: set of camera id to filter on
    returns query url
    
    gmaps_static_poly_main(cfg, s_id=3, cam_range=None) Filters on all scenario 3
    """
    if arg_in.polyline:
        print("Generating polyline plots...")
        if s_id is not None:
            for s in s_id:
                gmaps_static_poly_main(cfg, s_id=s, cam_range=c_id)
        else:
            gmaps_static_poly_main(cfg, s_id=s_id, cam_range=c_id)

    """Static marker plot method
    s_id: Scenario ID
    cam_range: set of camera id to filter on
    returns query url
    
    gmaps_static_marker_main(cfg, s_id=3, cam_range=None) Filters on all scenario 3
    """
    if arg_in.marker:
        print("Generating marker plots...")
        if s_id is not None:
            for s in s_id:
                gmaps_static_marker_main(cfg, s_id=s, cam_range=c_id)
        else:
            gmaps_static_marker_main(cfg, s_id=s_id, cam_range=c_id)

    """Waypoint/Route summary
    s_id: Scenario ID
    cam_range: set of camera id to filter
    gmaps_dir_sv_txt(cfg, s_id=3, cam_range=None) Filters on scenario 3
    """
    if arg_in.summary:
        print("Generating summary report...")
        if s_id is None:
            s_id = list(range(1, 6))
        for s in s_id:
            gmaps_dir_sv_txt(cfg, s_id=s, cam_range=c_id)

    if arg_in.cam_pair:
        if s_id is None:
            s_id = list(range(1, 6))
        for s in s_id:
            cam_link_main(cfg, s_id=s, cam_range=c_id)

    plot_pair()
    #camera_pair(cfg)

#
# def camera_pair(cfg):
#     waypoints_to_camera = cfg["wpmap"]
#     waypoints_pairs = cfg["pair"]
#     # cameras_set_from_waypoints_pairs # for every pair in waypoints pairs, get the camera set for point a & point b
#     print()
#     for pair in waypoints_pairs:
#         set_a = set(waypoints_to_camera[pair[0]])
#         set_b = set(waypoints_to_camera[pair[1]])
#         time = 0
#         if set_a == set_b:
#             time = 0  # no transition time
#         elif set_a.isdisjoint(set_b):
#             time = 1  # completely different
#         else:
#             time = 2
#

            # link other parts that are different (part is unique to other part) whcih have travel time
            # take the one that is common and link to all the ones that are different


    # figure out what the camera pairs look like.

    # both set completely same = intersection/ not transition time
    # both set completely different = normal/ there is transition time
    # both set got share values = to figure ot

def plot_pair():
    ts_str = []
    # March 5th, 2019
    time_year = 2019
    time_mon = 3
    # time_mday = 5

    for time_mday in range(5, 6):
        for time_hour in range(6, 18):  # Every hour
            for time_min in range(0, 60, 15):  # Every 15 minutes
                ts_str.append(
                    "./tmp/dir_ts_{d}_{m}_{y}_{h}_{min}.json".format(d=time_mday, m=time_mon, y=time_year, h=time_hour,
                                                                     min=time_min))
    traffic_duration_dict = {}  # dictionary has key as pairs and value as list of time
    normal_duration_dict = {}
    for json_file in ts_str:
        dict = json_read(json_file)
        for key, value in dict.items():
            list_normal = []  # array that stores list of time
            list_traffic = []
            duration_normal = value[0]['legs'][0]['duration']['value']
            duration_with_traffic = value[0]['legs'][0]['duration_in_traffic']['value']
            if key not in traffic_duration_dict:
                list_normal.append(duration_normal)
                list_traffic.append(duration_with_traffic)
                normal_duration_dict[key] = list_normal
                traffic_duration_dict[key] = list_traffic
            else:
                normal_duration_dict[key].append(duration_normal)
                traffic_duration_dict[key].append(duration_with_traffic)

    b = open('traffic_duration.csv', 'w')
    d = open('normal_duration.csv', 'w')
    a = csv.writer(b)
    c = csv.writer(d)
    traffic_data = []
    normal_data = []
    for key, value in traffic_duration_dict.items():
        y = value
        traffic_data.append(np.array(value))
        print()
        x = np.arange(6, 18, 0.25)
        plt.title('camera pair ' + key)
        plt.xlabel('clock time')
        plt.ylabel('travel time')
        plt.plot(x, y)
        plt.show()

    for key, value in normal_duration_dict.items():
        y = value
        normal_data.append(np.array(value))
        x = np.arange(6, 18, 0.25)
        plt.title('camera pair ' + key)
        plt.xlabel('clock time')
        plt.ylabel('travel time')
        plt.plot(x, y)
        plt.show()

    a.writerows(traffic_data)
    c.writerows(normal_data)
    b.close()
    print()




def config_init(cfgpath='./cfg/config.json'):
    """
    Performs any initializations using the configutation file.

    :param cfgpath: configuration file path
    :return:
    """
    cfg = json_read(cfgpath)
    return cfg


def cam_link_main(cfg, s_id=None, cam_range=None):
    """Main routine for generating camera link file
    :param cfg:
    :return:
    """
    link_d = traverse_cam_graph(cfg, s_id=s_id, cam_range=cam_range)
    write_cam_link(cfg, link_d, s_id=s_id)

    print()

def write_cam_link(cfg, link_dict, s_id=None):
    if s_id is not None:
        fn = './data/data_S{0}_autogen_camlink_{1}.txt'.format(s_id, cfg['date_ver'])

    else:
        fn = './data/data_autogen_camlink_{:s}.txt'.format(cfg['date_ver'])
    fid = open(fn, 'w')
    sorted_keys = list(link_dict.keys())
    sorted_keys.sort()

    fid.write(" ".join(["<camA>", "<camB>", "<time>", "<dist>", "<route>"]) + "\n")
    for k in sorted_keys:
        v = link_dict[k]    # List
        for e in v:     # For each edge/pair
            if e != -1:
                route_str = "{0},{1}".format(*e)
                route = cfg['directions'][route_str][0]

                t_dir = route["legs"][0]["duration"]["value"]  # Seconds
                d_dir = route["legs"][0]["distance"]["value"]  # Meters
                r_dir = route["overview_polyline"]["points"]
                # fid.write('{:d} {:d} {:d} {:d} {:d} {:d} {:s}\n'.format(*k, *e, t_dir, d_dir, r_dir)) # Write Edge too
                fid.write('{:d} {:d} {:d} {:d} {:s}\n'.format(*k, t_dir, d_dir, r_dir))
            else:
                # fid.write('{:d} {:d} {:d} {:d} {:d} {:d} {:s}\n'.format(*k, -1, -1, 0, 0, 'n/a')) # Write Edge too
                fid.write('{:d} {:d} {:d} {:d} {:s}\n'.format(*k, 0, 0, 'n/a'))
    fid.close()

    print(fn + " written!")

def traverse_cam_graph(cfg, s_id=None, cam_range=None):
    # Filter the waypoint set and camera set
    c_range = set(range(1, 41))  # 1 to 40

    # Filter
    if s_id is not None:
        c_range = set(cfg['scenario'][str(s_id)])
    if cam_range is not None:
        c_range = c_range.intersection(set(cam_range))

    # Determine the waypoint set
    wp_range = [cfg['cammap'][str(x)] for x in c_range]
    wp_range = set([int(e) for l in wp_range for e in l])

    # Waypoint set
    wp_open = list(wp_range)
    wp_open.sort()

    edgelist = [x for x in cfg['pair'] if all(elem in wp_open for elem in x)]   # Retrieves all edges within wp set

    # Generate waypoint -> Node hashmap,
    # Initialize Node objects
    node_map = dict()
    for wp in wp_open:
        cam_list = cfg['wpmap'][str(wp)]
        connections = list()
        for e in edgelist:
            if wp in e:
                connections.append(e)

        node_gen = Node(wp_id=wp, cam=cam_list, edges=connections)
        node_map[wp] = node_gen

    # Setup complete------------------------------

    """
    open_queue - Frontier set                          
    closed_queue - Explored set
    """
    open_queue = collections.deque([])  # Node queue
    closed_queue = collections.deque([])
    open_queue.append(node_map[wp_open[0]])

    cam_links = dict()

    # Main Loop
    while wp_open or edgelist:  # Remaining waypoint or edgelist
        if not open_queue:  # If open list empty, take waypoint at front of remaining wp and get its node
            print("Open queue empty! Getting next waypoint!")
            open_queue.append(node_map[wp_open[0]])
        # Remove queue head and push onto closed_queue
        n = open_queue.popleft()
        closed_queue.append(n)

        # Remove current waypoint from the remaining set
        wp_open.remove(n.waypoint_id)

        for e in n.edge_list:
            if e in edgelist:
                edgelist.remove(e)

            # Get next node
            n_next = node_map[[x for x in e if x != n.waypoint_id][0]]

            # Retrieve camera pair/links and cameras with same waypoint set
            cam_pairs, cam_same = get_camera_pairs(n.cam_list, n_next.cam_list)


            if cam_pairs:  # If paired
                for p in cam_pairs:  # For each sub-pair
                    if p in cam_links:  # if already a entry, append edge to camera link
                        cam_links[p].append(e)
                    else:
                        cam_links[p] = [e]
            elif cam_same:   # Same camera view
                for p in cam_same:
                    if p not in cam_links:
                        cam_links[p] = [-1]  # Represents no link (overlapping)

            # if cam_diff and len(cam_diff) % 2 == 0:
            #     cam_links.append([cam_diff, e])
            if n_next not in closed_queue and n_next not in open_queue:
                open_queue.append(n_next)
    return cam_links

def get_camera_pairs(setA, setB):
    set_intersect = setA & setB
    setA_u = setA - setB
    setB_u = setB - setA

    cam_pairs = list()

    if setA_u == setB_u:
        listA = list(setA)
        for i in range(0, len(listA)):
            for j in range(i+1, len(listA)):
                cam_pairs.append((listA[i], listA[j]))
        return [], set(cam_pairs)


    if setA_u and setB_u:
        for cA in setA_u:
            for cB in setB_u:
                cam_pairs.append((cA, cB))
    return set(cam_pairs), []



def gmaps_static_poly_main(cfg, s_id=None, cam_range=None):
    """Top-level main method for

    :param path_dict: gmaps query result
    :param flist: filter list (list of pairs)
    :return: query url, saves image to temp/
    """
    c_range = set(range(1, 41))  # 1 to 40

    # Filter
    if s_id is not None:
        c_range = set(cfg['scenario'][str(s_id)])
    if cam_range is not None:
        c_range = c_range.intersection(set(cam_range))

    # Determine the waypoint set
    wp_range = [cfg['cammap'][str(x)] for x in c_range]
    wp_range = set([int(e) for l in wp_range for e in l])

    # Filter pair list
    # flist = [[0, 1], [2, 3]...] filters on pairs 0-1 and 2-3 ...
    flist = ["{0},{1}".format(o, d) for o, d in cfg['pair'] if o in wp_range and d in wp_range]

    path_str = list()
    path_dict = cfg['directions']
    for k in path_dict:
        if flist and k in flist:
            v = path_dict[k]
            path_str.append(v[0]['overview_polyline']['points'])

    fullquery = gmaps_static_poly_img(path_str)
    return fullquery


def gmaps_static_poly_img(path_list):
    baseURL = 'https://maps.googleapis.com/maps/api/staticmap?'

    apikey = 'AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY'
    mapsize = [640, 640]
    mapstyle = 'style=feature:all|element:labels|visibility:off'
    sz = None  # {None, small, mid, tiny}
    mapscale = 2
    mapzoom = gbl_gmap_zoom
    # mapcenter = [42.496553, -90.684858]

    gmap_key = 'key={0}'.format(apikey)

    gmap_size = 'size={0}x{1}'.format(*mapsize)
    gmap_scale = 'scale={0}'.format(mapscale)
    gmap_zoom = 'zoom={0}'.format(mapzoom)
    # gmap_center = 'center={0},{1}'.format(*mapcenter)
    gmap_paths = ["path=color:{0}%7Cweight:{1}%7Cenc:{2}".format("0x0000ff80", 3, x) for x in path_list]

    fullquery = baseURL + "&".join([gmap_size, mapstyle, gmap_scale, gmap_zoom, *gmap_paths, gmap_key])
    retrieve_image(fullquery)
    return fullquery


def gmaps_static_marker_main(cfg, s_id=None, cam_range=None):
    # Defaults
    s_type = ['train', 'test', 'train', 'train', 'test']
    # wp_range = [int(k) for k in cfg['gps']]
    c_range = set(range(1, 41))  # 1 to 40

    # Filter
    p_type = 'n/a'
    s_name = None
    if s_id is not None:
        c_range = set(cfg['scenario'][str(s_id)])
        p_type = s_type[s_id - 1]
        s_name = "S{:d}".format(s_id)
    if cam_range is not None:
        c_range = c_range.intersection(set(cam_range))

    # Determine the waypoint set
    wp_range = [cfg['cammap'][str(x)] for x in c_range]
    wp_range = set([int(e) for l in wp_range for e in l])

    gpslist = [[wp, *cfg['gps'][str(wp)]] for wp in wp_range]
    fullquery = gmaps_static_marker_img(gpslist)
    return fullquery


def gmaps_static_marker_img(gps_list):
    baseURL = 'https://maps.googleapis.com/maps/api/staticmap?'
    apikey = 'AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY'
    mapsize = [640, 640]
    mapstyle = 'style=feature:all|element:labels|visibility:off'
    sz = None  # {None, small, mid, tiny}
    mapscale = 2
    mapzoom = gbl_gmap_zoom
    # mapcenter = [42.496553, -90.684858]

    gmap_key = 'key={0}'.format(apikey)

    gmap_size = 'size={0}x{1}'.format(*mapsize)
    gmap_scale = 'scale={0}'.format(mapscale)
    gmap_zoom = 'zoom={0}'.format(mapzoom)
    gmap_markers = "&".join(genSimpleMarkerQuery2(gps_list))
    # gmap_center = 'center={0},{1}'.format(*mapcenter)

    fullquery = baseURL + "&".join([gmap_size, gmap_scale, gmap_zoom, gmap_markers, gmap_key])
    retrieve_image(fullquery)
    return fullquery


def genSimpleMarkerQuery2(gpslist):
    colororder = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white']

    mquery = list()
    while gpslist:
        waypoint = gpslist[0]
        gps = waypoint[1:3]

        cam_color = (int(waypoint[0]) % 100) // 10
        cam_lbl = chr(ord('0') + int(waypoint[0]) % 10)

        marker_str = "markers="
        marker_color = "color:{0}".format(colororder[cam_color])
        marker_lbl = "label:{0}".format(cam_lbl)
        marker_loc = "{0:.06f},{1:.06f}".format(*gps)
        marker_size = "size:{0}".format('mid')

        marker_str += "%7C".join([marker_lbl, marker_color, marker_size, marker_loc])

        mquery.append(marker_str)
        gpslist = gpslist[1:]
    return mquery


def gmaps_dir_sv_txt(cfg, s_id=None, cam_range=None):
    """Main routine to generate summary text file
    :param cfg: configuration data
    :return:
    """
    s_type = ['train', 'test', 'train', 'train', 'test']

    # Defaults
    # wp_range = [int(k) for k in cfg['gps']]
    c_range = set(range(1, 41))  # 1 to 40

    # Filter
    p_type = 'n/a'
    s_name = None
    if s_id is not None:
        c_range = set(cfg['scenario'][str(s_id)])
        p_type = s_type[s_id-1]
        s_name = "S{:d}".format(s_id)
    if cam_range is not None:
        c_range = c_range.intersection(cam_range)

    # Determine the waypoint set
    wp_range = [cfg['cammap'][str(x)] for x in c_range]
    wp_range = set([int(e) for l in wp_range for e in l])

    output_txt = list()
    for orig, dest in cfg['pair']:
        if orig in wp_range and dest in wp_range:
            data = cfg['directions']["{0},{1}".format(orig, dest)][0]
            t_dir = data["legs"][0]["duration"]["value"]  # Seconds
            d_dir = data["legs"][0]["distance"]["value"]  # Meters
            r_dir = data["overview_polyline"]["points"]

            res_str = "{:d} {:d} {:s} {:.06f} {:.06f} {:.06f} {:.06f} {:d} {:d} {:s}\n".format(orig, dest, p_type,
                                                                                          *cfg['gps'][str(orig)], *cfg['gps'][str(dest)],
                                                                                          t_dir, d_dir, r_dir)
            output_txt.append(res_str)

    # hash_enc = hashlib.sha224(str.encode(cfg['hash'])).hexdigest()
    # if s_name is not None:
    #     save_fn = "./data/{0}_data_autogen_".format(s_name) + hash_enc + ".txt"
    # else:
    #     save_fn = "./data/{0}_data_autogen_".format(str(uuid.uuid4().hex)[:4]) + hash_enc + ".txt"
    hash_enc = hashlib.sha224(str.encode(cfg['hash'])).hexdigest()
    if s_name is not None:
        save_fn = "./data/data_{0}_autogen_".format(s_name) + cfg['date_ver'] + ".txt"
    else:
        save_fn = "./data/data_{0}_autogen_".format(str(uuid.uuid4().hex)[:4]) + cfg['date_ver'] + ".txt"

    fid = open(save_fn, "w")
    [fid.write(s) for s in output_txt]
    fid.close()


def gmaps_dir_extract(gps):
    """Google Maps Directions API single query
    :param gps: list of gps pair [origin, dest]
    :return: Travel duration, distance, encoded polyline
    """
    # Define directions parameters
    mode = "driving"  # Driving
    units = "metric"  # Metric Units
    language = "en"  # English
    departure_time = "now"
    traffic_model = "best_guess"

    o, d = gps

    gmaps = googlemaps.Client(key='AIzaSyAcrr9cNHjm1IIgt1txjG9TAL-r5_Bx5TY')
    query = gmaps.directions(origin=o, destination=d,
                             mode=mode, units=units, language=language, departure_time=departure_time,
                             traffic_model=traffic_model)
    travel_dur = query[0]["legs"][0]["duration"]["value"]  # Seconds
    distance = query[0]["legs"][0]["distance"]["value"]  # Meters
    polyline = query[0]["overview_polyline"]["points"]

    return travel_dur, distance, polyline, query


def json_read(fp):
    """Reads stored json data and converts to a Python dict structure

    :param fp: json file path
    :type fp: string
    :return: json data as dictionary
    """
    if os.path.splitext(fp)[1] != '.json':
        raise Exception("Invalid file extension to json_reads")
    with open(fp, "r") as f:
        rd_data = f.read()
    rd_data = json.loads(rd_data)
    return rd_data


def json_write(svpath, wr_data):
    """Writes Python dict structure to json file specified by svpath.

    :param svpath:
    :param wr_data:
    :return:
    """
    svpath = check_ext(svpath, '.json')
    with open(svpath, "w") as f:
        f.write(json.dumps(wr_data))
    print("{0} written.".format(svpath))


def check_ext(path, targ_ext):
    return os.path.splitext(path)[0] + targ_ext


def retrieve_image(query, svpath=None):
    """Retrieves an image from a static Google Maps API call and saves it using a
    randomized filename in a temp directory (unless save path is otherwise specified)

    :param query: url query
    :type query: basestring

    :param svpath:
    :return:
    """
    if svpath is None:  # Generate randomized save filename
        svpath = "tmp/" + str(uuid.uuid4().hex) + ".png"

    print("Writing to {0}".format(svpath))
    ret = urllib.request.urlretrieve(query)   # Retrieve image
    cv2.imwrite(svpath, cv2.imread(ret[0]))   # Save image


"""
Setup config files
Do not modify
"""
def config_setup():
    """
    Creates config.json
    Requires:
        Link Matrix (.csv)
        Waypoint Description - GPS Locations (.txt)
        Camera to Waypoint mapping (.txt)
    :return:
    """
    hash_str = "Hello World!"  # 2/24/2019, Kelvin Lin

    linkmat_path = "./cfg/modified_waypoints_friendly_matrix_v4.csv"   # Link Matrix
    waypoint_path = "./cfg/waypoints_1_40_and_S5_v2.txt"                      # Waypoint Description
    cam_wp_path = "./cfg/cam_to_waypoint_v4.txt"                       # Camera to Waypoint Map

    train_set_1 = list(range(1, 6))                                     # Scenario 1, Training, Cameras 1 to 5
    test_set_2 = list(range(6, 10))                                     # Scenario 2, Testing,  Cameras 6 to 9
    train_set_3 = list(range(10, 16))                                   # Scenario 3, Training, Cameras 10 to 15
    train_set_4 = list(range(16, 41))                                   # Scenario 4, Training, Cameras 16 to 40
    test_set_5 = list([10, *range(16, 30), *range(33, 37)])             # Scenario 5, Testing,  Cameras 10, 16 to 30, 33 to 37

    train_set = [*train_set_1, *train_set_3, *train_set_4]  # Training Set
    test_set = [*test_set_2, *test_set_5]  # Testing

    with open(linkmat_path, 'r') as f:
        lmdata = f.read().splitlines()
    lm_mat = [x.split(sep=',') for x in lmdata]  # Matrix

    with open(waypoint_path, 'r') as f:
        wpdata = f.read().splitlines()
    wpdata = [np.fromstring(x, sep=' ') for x in wpdata]
    wp = dict((int(k[0]), k[2:4].tolist()) for k in wpdata)

    with open(cam_wp_path, 'r') as f:
        camdata = f.read().splitlines()
    camdata =[np.fromstring(x, sep=' ') for x in camdata]
    cam_dict = dict((int(x[0]), x[1:].tolist()) for x in camdata)

    wp_dict = reverse_dict(cam_dict)

    scenario = {1: train_set_1, 2: test_set_2, 3: train_set_3, 4: train_set_4, 5: test_set_5}

    lmpairs, lm_copy = linkmat_to_pair(lm_mat)           # Waypoint Pairs
    pair_json = gmaps_dir_json(lmpairs, wp)     # Google Maps Directions API results
    copy_json = gmaps_dir_json(lm_copy, wp)

    # Resolve duplicate pairs -> Keep the routes with lower travel time
    swap_ct = 0
    for k in lm_copy:
        k_str = "{0},{1}".format(*k)
        k_rev = "{0},{1}".format(k[1], k[0])
        if k_rev in pair_json:
            # If copy is shorter (time), swap
            if copy_json[k_str][0]['legs'][0]['duration']['value'] < pair_json[k_rev][0]['legs'][0]['duration']['value']:
                # Swap dictionary entry
                pair_json.pop(k_rev)
                pair_json[k_str] = copy_json[k_str]

                # Update pair list
                lmpairs.remove([k[1], k[0]])
                lmpairs.append(k)

                swap_ct += 1
    print("Performed {0} swaps.".format(swap_ct))

    config_dict = dict()
    config_dict['directions'] = pair_json
    config_dict['pair'] = lmpairs
    config_dict['gps'] = wp
    config_dict['cammap'] = cam_dict
    config_dict['wpmap'] = wp_dict
    config_dict['scenario'] = scenario
    config_dict['hash'] = hash_str
    config_dict['date_ver'] = str(datetime.datetime.now().strftime("%Y_%m_%d") + "_{:d}".format(int(time.mktime(time.gmtime()))) )
    json_write('./cfg/config.json', config_dict)
    exit(0)


def gmaps_dir_json(lmpairs, wplist):
    res = dict()

    i = 0
    for o, d in lmpairs:
        # Travel Duration(t), Travel Distance(d), Encoded Polyline/Route(r)
        _, _, _, query = gmaps_dir_extract([wplist[o], wplist[d]])
        res["{0},{1}".format(o, d)] = query
    # json_write("./data/dir_config_auto_"+str(uuid.uuid4().hex)+".json", res)

    print()
    return res


def linkmat_to_pair(lm_mat):
    """
    Reads the link matrix and the waypoint descriptions to generate the link matrix pairs and GPS coordinates
    :param lmpath:
    :param wppath:
    :param wr_file:
    :return:
    """

    lm_header = [int(x) for x in lm_mat[0][1:]]

    data_mat_num = []
    for l in lm_mat[1:]:
        temp = list()
        for e in l[1:]:
            if e == '':
                temp.append(0)
            else:
                temp.append(int(e))
        data_mat_num.append(temp)
    data_mat_nz = np.nonzero(data_mat_num)

    lm_pairs = list()
    lm_copies = list()
    for x, y in zip(*data_mat_nz):
        if [y, x] not in lm_pairs:
            lm_pairs.append([x, y])
        else:
            lm_copies.append([x, y])
    lm_pairs = [[lm_header[x], lm_header[y]] for x, y in lm_pairs]
    lm_copies = [[lm_header[x], lm_header[y]] for x, y in lm_copies]

    return lm_pairs, lm_copies


def dump_wp_main(data_pairs, wplist):
    fid = open("data/wp_gps_pairs_auto_v3.txt", "w")
    fid2 = open("data/wp_pairs_auto_v3.txt", "w")
    for x, y in data_pairs:
        fid.write("{0} {1} {2} {3}\n".format(*wplist[x], *wplist[y]))
        fid2.write("{0} {1}\n".format(x, y))
    fid.close()
    fid2.close()


def reverse_dict(d):
    rev_d = dict()
    for k in d:
        v = map(int, d[k])
        for v_i in v:
            if v_i in rev_d.keys():
                rev_d[v_i].extend([k])
            else:
                rev_d[v_i] = [k]
    return rev_d


if __name__ == "__main__":
    main(sys.argv[1:])