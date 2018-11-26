"""Intersection reach analysis.

Author: Chen Feng
Last updated on Nov. 21, 2018
"""

from __future__ import division
from dijkstra import Dijkstra
import numpy as np
import pandas as pd
import os
import time
from pyx import *


# __________Reformat data__________

def get_EV_id_matrix(na_x1, na_y1, na_x2, na_y2):
    """Produce the edge-vertex-ID matrix for the undirected graph generated from the input map.

    :param na_x1: a 1-d numpy array listing the x-coordinate of the start point of the undirected edge/segment
    :param na_y1: a 1-d numpy array listing the y-coordinate of the start point of the undirected edge/segment
    :param na_x2: a 1-d numpy array listing the x-coordinate of the end point of the undirected edge/segment
    :param na_y2: a 1-d numpy array listing the y-coordinate of the end point of the undirected edge/segment
    :return: a 2-d numpy array listing the start and end point IDs of each undirected edge, a 1-d numpy array
            listing x-y coordinate tuple of the start point of each edge, and a 1-d numpy array listing
            x-y coordinate tuple of the end point of each edge
    """
    # Produce the coordinates of the start points of segments and
    # convert to a numpy array of tuples in the form of [(x1, y1), ...]
    na_pts_start = (np.array(zip(na_x1, na_y1), dtype=[('x1', float), ('y1', float)])).astype(tuple)
    list_pts_start = na_pts_start.tolist()
    # Produce the coordinates of the end pints of segments and
    # convert to a numpy array of tuples in the form of [(x2, y2), ...]
    na_pts_end = (np.array(zip(na_x2, na_y2), dtype=[('x2', float), ('y2', float)])).astype(tuple)
    list_pts_end = na_pts_end.tolist()
    # Extract distinct vertices from the point list
    list_pts = na_pts_start.tolist() + na_pts_end.tolist()
    set_pts = list(set(list_pts))
    list_pt_id = range(len(set_pts))
    # Create an edge-vertex matrix with the column 'na_edge_pts_1' indicating the start vertices, and
    # the column 'na_edge_pts_2' indicating the end vertices.
    na_edge_pts_1 = np.zeros((len(na_x1)), dtype=int) - 1  # create a numpy array filled with -1
    for i in range(len(list_pt_id)):
        indices = [ndx for ndx, x in enumerate(list_pts_start) if x == set_pts[i]]
        for j in indices:
            na_edge_pts_1[j] = i

    na_edge_pts_2 = np.zeros((len(na_x1)), dtype=int) - 1  # create a numpy array filled with -1
    for i in range(len(list_pt_id)):
        indices = [ndx for ndx, x in enumerate(list_pts_end) if x == set_pts[i]]
        for j in indices:
            na_edge_pts_2[j] = i

    na_edge_pts = np.column_stack((na_edge_pts_1, na_edge_pts_2))

    return na_edge_pts, na_pts_start, na_pts_end


def create_x_adj_list(na_edge_pts, deg_threshold):
    """Create an adjacency list for the undirected graph generated from the initial map.

    :param na_edge_pts: a 2-d numpy array listing the start and end point IDs of each edge
    :param deg_threshold: the minimum degree of a vertex (i.e., an endpoint) to be considered as a street intersection
    :return: an adjacency list, e.g., G = [{12: 0, 1: 1, 24: 1}, {0: 1, 24:1, 2: 1, 16:1}, ...]
            indicates that the edge 0 is adjacent with edges 12, 1, and 24, and the cross distances from
            those edges are G[0][12] = 0, G[0][1] = 1, and G[0][24] = 1.
    """
    adj_list = []
    for i in range(len(na_edge_pts)):
        adj_list.append({})
        # extract the start point of the current edge
        v1 = na_edge_pts[i, 0]
        # find the incident edges of the start point of the current edge
        t = np.where(na_edge_pts == v1)
        v1_deg = len(t[0])
        for j in range(len(t[0])):
            if t[0][j] != i:
                if v1_deg >= deg_threshold:
                    adj_list[i][int(t[0][j])] = 1
                else:
                    adj_list[i][int(t[0][j])] = 0
        # extract the end point of the current edge
        v2 = na_edge_pts[i, 1]
        # find the incident edges of the end point of the current edge
        t = np.where(na_edge_pts == v2)
        v2_deg = len(t[0])
        for j in range(len(t[0])):
            if t[0][j] != i:
                if v2_deg >= deg_threshold:
                    adj_list[i][int(t[0][j])] = 1
                else:
                    adj_list[i][int(t[0][j])] = 0

    return adj_list


# __________Run intersection-reach analysis__________

def x_reach(edge_id, max_cross, adj_list, na_seg_len):
    """Conduct intersection reach analysis.

    :param edge_id: the ID of the source edge
    :param max_cross: the maximum number of intersections allowed to cross
    :param adj_list: an adjacency list, e.g., G = [{12: 0, 1: 1, 24: 1}, {0: 1, 24:1, 2: 1, 16:1}, ...]
                     indicates that the edge 0 is adjacent with edges 12, 1, and 24, and the cross distances from
                     those edges are G[0][12] = 0, G[0][1] = 1, and G[0][24] = 1.
    :param na_seg_len: a list listing the length of each line segment
    :return: the total street length accessible within max_cross intersections and
            the list of lines that can be reached
    """
    D, P = Dijkstra(adj_list, edge_id)
    # find all the edges that are no more than max_cross intersections away from the source edge
    list_reached_edges = [k for k, v in D.items() if v <= max_cross]
    # compute the total length of the reached edges
    total_len = 0
    for i in list_reached_edges:
        total_len += na_seg_len[i]

    return total_len, list_reached_edges


def x_reach_all_pairs(max_cross, adj_list, na_seg_len):
    """Conduct intersection reach analysis for all lines.

    :param max_cross: the maximum number of intersections allowed to cross
    :param adj_list: an adjacency list, e.g., G = [{12: 0, 1: 1, 24: 1}, {0: 1, 24:1, 2: 1, 16:1}, ...]
                     indicates that the edge 0 is adjacent with edges 12, 1, and 24, and the cross distances from
                     those edges are G[0][12] = 0, G[0][1] = 1, and G[0][24] = 1.
    :param na_seg_len: a list listing the length of each line segment
    :return: a list listing the intersection reach for each line segment
    """
    list_reach = []
    num_lines = len(na_seg_len)
    for e in range(num_lines):
        reach, list_edges = x_reach(e, max_cross, adj_list, na_seg_len)
        list_reach.append(reach)

    return list_reach


def draw_x_reach_edges(eid, list_edge_id, na_e_pt_start, na_e_pt_end, file_dir, file_name="x_reach_snapshot",
                       stroke_width=0.02, scale_factor=1 / 50):
    """Visualize the result of intersection reach analysis.

    :param eid: the ID of the edge from which the reach analysis starts
    :param list_edge_id: a list listing the IDs of the line segments that have been reached
    :param na_e_pt_start: a 1-d numpy array listing x-y coordinate tuple of the start point of each line
    :param na_e_pt_end: a 1-d numpy array listing x-y coordinate tuple of the end point of each line
    :param file_dir: a string representing the file directory in which to store the exported drawing
    :param file_name: the name of the PDF file to be exported to
    :param stroke_width: the width of the stroke for drawing lines
    :param scale_factor: the scale factor used to scale the input coordinates
    """
    os.chdir(file_dir)
    # Set canvas for drawing
    c = canvas.canvas()
    num_edges = len(na_e_pt_start)

    # Draw the whole grid (i.e., the initial map) first
    for i in range(num_edges):
        line = path.line(na_e_pt_start[i][0] * scale_factor, na_e_pt_start[i][1] * scale_factor,
                         na_e_pt_end[i][0] * scale_factor, na_e_pt_end[i][1] * scale_factor)
        c.stroke(line, [style.linewidth(stroke_width), color.rgb.black])

    # Draw the reached edges in list_edge_id
    for e in list_edge_id:
        line = path.line(na_e_pt_start[e][0] * scale_factor, na_e_pt_start[e][1] * scale_factor,
                         na_e_pt_end[e][0] * scale_factor, na_e_pt_end[e][1] * scale_factor)
        c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])

    # Draw a red circle at the midpoint of the edge from which to start
    circle_center_x = (na_e_pt_start[eid][0] + na_e_pt_end[eid][0]) / 2
    circle_center_y = (na_e_pt_start[eid][1] + na_e_pt_end[eid][1]) / 2
    circle = path.circle(circle_center_x * scale_factor, circle_center_y * scale_factor, 8 * stroke_width)
    c.stroke(circle, [deco.filled([color.rgb.black])])

    c.writePDFfile(file_name)


if __name__ == '__main__':
    # Document the time at which the script starts running
    localtime = time.asctime(time.localtime(time.time()))
    print "Start Time :", localtime + "\n"

    # Change working directory
    directory = r"C:\_SoftwareDevelopment\Grasshopper\GhPython_PatternGeneration\data_RealExamples\Apt"
    os.chdir(directory)
    csv_file = "test_Apt.csv"
    # Set the maximum number of intersections allowed to travel
    max_crossings = 3
    # Set the vertex degree threshold
    degree_threshold = 3
    # Set the start edge ID
    start_edge = 125

    # Read in the data and store them as a pandas DataFrame
    df = pd.read_csv(csv_file)

    x1, y1, x2, y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_segment_length = df['seg_len'].values

    na_edge_points, na_points_start, na_points_end = get_EV_id_matrix(x1, y1, x2, y2)
    adjacency_list = create_x_adj_list(na_edge_points, degree_threshold)
    xr, reached_edgs = x_reach(start_edge, max_crossings, adjacency_list, na_segment_length)
    print xr
    # print x_reach_all_pairs(max_crossings, adjacency_list, na_segment_length)

    # Document the time at which the script finishes running
    localtime = time.asctime(time.localtime(time.time()))
    print "\nEnd Time :", localtime

    # Change the directory to save the PDF file
    directory = "C:/Users/cfeng/Desktop/Outbox"
    # Highlight the reached edges on the initial map and export the drawing to a PDF file
    draw_x_reach_edges(start_edge, reached_edgs, na_points_start, na_points_end, directory, stroke_width=0.02,
                       scale_factor=1 / 50)
