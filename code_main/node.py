class Node:
    def __init__(self, wp_id, cam, edges):
        self.waypoint_id = wp_id
        self.cam_list = set(cam)
        self.edge_list = edges

    # def __str__(self):
    #     return "Waypoint {0}".format(self.waypoint_id)

    #
    # def search(self, nodes_to_visit):
    #     """
    #         Search nodes' connections starting from  the node
    #
    #         """
    #     nodes_to_visit = deque(nodes_to_visit)
    #     node_visited = set([])
    #     known_nodes = set([])
    #     while nodes_to_visit.size() > 0:  # loop through all waypoints
    #         current = nodes_to_visit.dequeue()
    #         neighbors = current.get_neighbor()  # get a list of current waypoint's neighbor
    #         for neighbor in neighbors:
    #             if neighbor not in known_nodes:  # if neighbor is unknown
    #                 nodes_to_visit.add(neighbor)
    #                 known_nodes.add(neighbor)  # mark neighbor is known
    #
    #             node_visited.add(neighbor)  # mark as visited
    #
    #
    # def get_neighbor(self):
    #     neighbors = set([edges_out])
    #     return neighbors
    #
