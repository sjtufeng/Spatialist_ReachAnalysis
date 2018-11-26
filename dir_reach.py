"""Directional reach analysis.

Authors: Chen Feng, Wenwen Zhang
Last updated on Nov. 21, 2018
"""

from __future__ import division
import os
import time
import math
import numpy as np
import pandas as pd
from pyx import *
from dijkstra import Dijkstra


# __________Reformat data__________

def get_directed_EV_id_matrix(na_x1, na_y1, na_x2, na_y2):
    """Produce the edge-vertex-ID matrix for the directed graph converted from the initial map by appending flipped edges.

    :param na_x1: a 1-d numpy array listing the x-coordinate of the start point of the undirected edge/segment
    :param na_y1: a 1-d numpy array listing the y-coordinate of the start point of the undirected edge/segment
    :param na_x2: a 1-d numpy array listing the x-coordinate of the end point of the undirected edge/segment
    :param na_y2: a 1-d numpy array listing the y-coordinate of the end point of the undirected edge/segment
    :return: a 2-d numpy array listing the starting and ending point IDs of each directed edge (with the flipped edges
            appended to the original edges), a 1-d numpy array listing x-y coordinate tuple of the start point of
            each directed edge, and a 1-d numpy array listing x-y coordinate tuple of the end point of
            each directed edge
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

    # Create the edge-vertex matrix for the flipped edges and concatenate two matrices
    flipped_na_edge_pts = na_edge_pts.copy()
    flipped_na_edge_pts[:, 0], flipped_na_edge_pts[:, 1] = na_edge_pts[:, 1], na_edge_pts[:, 0]
    directed_na_edge_pts = np.concatenate((na_edge_pts, flipped_na_edge_pts))

    # Create the coordinates for the flipped edges and concatenate with the original coordinates
    directed_na_pts_start = np.concatenate((na_pts_start, na_pts_end))
    directed_na_pts_end = np.concatenate((na_pts_end, na_pts_start))

    return directed_na_edge_pts, directed_na_pts_start, directed_na_pts_end


# __________Create adjacency list__________

# ----------helper functions starts----------

def get_radian_angle_btwn_vecs(vec1_x, vec1_y, vec2_x, vec2_y):
    """Compute the angle between two vectors (vec1, vec2) in radians.

    :param vec1_x: the x-coordinate of vec1
    :param vec1_y: the y-coordinate of vec1
    :param vec2_x: the x-coordinate of vec2
    :param vec2_y: the y-coordinate of vec2
    :return: the angle between vec1 and vec2
    """
    test = (vec1_x * vec2_x + vec1_y * vec2_y) / math.sqrt(vec1_x ** 2 + vec1_y ** 2) / math.sqrt(
        vec2_x ** 2 + vec2_y ** 2)
    if test > 1:
        test = 1
    if test < -1:
        test = -1
    rad = math.acos(test)

    return rad


def get_angle_btwn_vecs(vec1_x, vec1_y, vec2_x, vec2_y):
    """Compute the angle between two vectors (vec1, vec2) in degrees.

    :param vec1_x: the x-coordinate of vec1
    :param vec1_y: the y-coordinate of vec1
    :param vec2_x: the x-coordinate of vec2
    :param vec2_y: the y-coordinate of vec2
    :return: the angle between vec1 and vec2
    """
    rad = get_radian_angle_btwn_vecs(vec1_x, vec1_y, vec2_x, vec2_y)
    return radian_to_degree(rad)


def radian_to_degree(radian):
    """Convert radians to degrees.

    :param radian: the angle in radians
    :return: the angle in degrees
    """
    deg = radian * 180 / math.pi

    return deg


# ----------helper functions ends----------

def create_directed_adj_list(directed_na_edge_pts, directed_na_pts_start, directed_na_pts_end, ang_threshold):
    """Create an adjacency list for the directed graph converted from the initial map by appending flipped edges.

    The distance cost between two edges is defined by the number of direction changes involved in traveling from one to
    the other.

    :param directed_na_edge_pts: a 2-d numpy array listing the starting and ending point IDs of each directed edge
                                (with the flipped edges appended to the original edges)
    :param directed_na_pts_start: a 1-d numpy array listing x-y coordinate tuple of the start point of
                                each directed edge
    :param directed_na_pts_end: a 1-d numpy array listing x-y coordinate tuple of the end point of
                                each directed edge
    :param ang_threshold: the angle threshold used to determine whether to count as a direction change
    :return: an adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
            the edge 0 is incident with the edges 1, 26, and 50, and directional distances from those edges are
            G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    """
    adj_list = []
    for i in range(len(directed_na_edge_pts)):
        # extract the end point of  the current directed edge
        endpt = directed_na_edge_pts[i, 1]
        # check which edges have the start point same as the end point of the current edge
        t = np.where(directed_na_edge_pts[:, 0] == endpt)
        adj_list.append({})
        vec1_x = directed_na_pts_end[i][0] - directed_na_pts_start[i][0]
        vec1_y = directed_na_pts_end[i][1] - directed_na_pts_start[i][1]
        for j in range(len(t[0])):
            vec2_x = directed_na_pts_end[t[0][j]][0] - directed_na_pts_start[t[0][j]][0]
            vec2_y = directed_na_pts_end[t[0][j]][1] - directed_na_pts_start[t[0][j]][1]
            rad = get_radian_angle_btwn_vecs(vec1_x, vec1_y, vec2_x, vec2_y)
            deg = radian_to_degree(rad)
            adj_list[i][int(t[0][j])] = deg

    # Update the adjacency list based on the predefined angle threshold
    for i in range(len(adj_list)):
        for k, v in adj_list[i].items():
            if v > ang_threshold:
                adj_list[i][k] = 1
            else:
                adj_list[i][k] = 0

    return adj_list


# __________Run directional-reach analysis__________

def min_directional_distance(edge_id, adj_list):
    """Find the shortest directional distance from the specified edge to all the other edges in the graph.

    :param edge_id: the ID of the source edge
    :param adj_list: the adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
                    the edge 0 is incident with the edges 1, 26, and 50, and distances from those edges are
                    G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    :return: a dictionary of shortest directional distances from the specified edge
    """
    num_lines = len(adj_list) // 2  # the total number of line segments (i.e., undirected edges)
    source_id_1 = edge_id
    source_id_2 = edge_id + num_lines
    # Run dijkstra twice, respectively based on the two directed source edges
    D1, P1 = Dijkstra(adj_list, source_id_1)
    D2, P2 = Dijkstra(adj_list, source_id_2)
    # Compare D1 and D2 to get the shortest distance for each directed edge
    dict_min_dist = {}
    for k in D1:
        if k in D2:
            dict_min_dist[k] = min(D1[k], D2[k])
        else:
            dict_min_dist[k] = D1[k]
    for k in D2:
        if k in D1:
            continue
        else:
            dict_min_dist[k] = D2[k]
    # Compare the pairs of flipped edges to find the real shortest distance for the line segment
    dict_real_min_dist = {}
    for k, v in dict_min_dist.items():
        # find the ID of the opposite edge, assuming k is no smaller than the total number of line segments
        i = k - num_lines
        # find the ID of the opposite edge, assuming k is smaller than the total number of line segments
        j = k + num_lines
        if i >= 0:
            if i in dict_min_dist:
                dict_real_min_dist[i] = min(dict_min_dist[i], v)
            else:
                dict_real_min_dist[i] = v
        else:
            if j in dict_min_dist:
                dict_real_min_dist[k] = min(v, dict_min_dist[j])
            else:
                dict_real_min_dist[k] = v

    return dict_real_min_dist


def d_reach(edge_id, max_dc, adj_list, na_seg_len):
    """Conduct directional reach analysis.

    :param edge_id: the ID of the source edge
    :param max_dc: the maximum number of direction changes allowed
    :param adj_list: the adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
                    the edge 0 is incident with the edges 1, 26, and 50, and distances from those edges are
                    G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    :param na_seg_len: a list showing the length of each line segment
    :return: the total street length accessible within max_dc direction changes and
            the list of lines that can be reached
    """
    dict_dd = min_directional_distance(edge_id, adj_list)
    # Extract the edges that are no more than max_dc direction changes away from the source edge
    list_reached_edges = [k for k, v in dict_dd.items() if v <= max_dc]
    # Compute the total length of the reached edges
    total_len = 0
    for i in list_reached_edges:
        total_len += na_seg_len[i]

    return total_len, list_reached_edges


def get_DDL(edge_id, adj_list, na_seg_len):
    """Compute the directional distance per length (DDL) for the specified edge.

    :param edge_id: the ID of the source edge
    :param adj_list: the adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
                    the edge 0 is incident with the edges 1, 26, and 50, and distances from those edges are
                    G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    :param na_seg_len: a list showing the length of each line segment
    :return: the value of DDL
    """
    dict_dd = min_directional_distance(edge_id, adj_list)
    total_len_x_dd = 0
    for k, v in dict_dd.items():
        total_len_x_dd += na_seg_len[k] * v
    ddl = total_len_x_dd / np.sum(np.asarray(na_seg_len))

    return ddl


def d_reach_all_pairs(max_dc, adj_list, na_seg_len):
    """Conduct directional reach analysis for all lines.

    :param max_dc: the maximum number of direction changes allowed
    :param adj_list: the adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
                    the edge 0 is incident with the edges 1, 26, and 50, and distances from those edges are
                    G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    :param na_seg_len: a list showing the length of each line segment
    :return: a list listing the directional reach for each line segment
    """
    list_reach = []
    num_lines = len(na_seg_len)
    for e in range(num_lines):
        reach, list_edges = d_reach(e, max_dc, adj_list, na_seg_len)
        list_reach.append(reach)

    return list_reach


def get_DDL_all_pairs(adj_list, na_seg_len):
    """Compute DDL for all lines.

    :param adj_list: the adjacency list, e.g., G = [{1: 0, 26: 1, 50: 1}, {16: 1, 2: 0, 27: 1}, ...] indicates that
                    the edge 0 is incident with the edges 1, 26, and 50, and distances from those edges are
                    G[0][1] = 0, G[0][26] = 1, and G[0][50] = 1.
    :param na_seg_len: a list showing the length of each line segment
    :return: a list listing the DDL for each line segment
    """
    list_ddl = []
    num_lines = len(adj_list) // 2
    for e in range(num_lines):
        ddl = get_DDL(e, adj_list, na_seg_len)
        list_ddl.append(ddl)

    return list_ddl


def get_mean_DDL(adj_list, na_seg_len):
    """Compute the length-weighted mean of DDL."""
    list_ddl = get_DDL_all_pairs(adj_list, na_seg_len)
    return sum([ddl * na_seg_len[i] / np.sum(np.asarray(na_seg_len)) for i, ddl in enumerate(list_ddl)])


def draw_d_reach_edges(eid, list_edge_id, na_e_pt_start, na_e_pt_end, file_dir, file_name="dir_reach_snapshot",
                       stroke_width=0.02, scale_factor=1 / 50):
    """Visualize the result of directional reach analysis.

    :param eid: the ID of the edge from which the reach analysis starts
    :param list_edge_id: a list listing the IDs of the undirected edges that have been reached
    :param na_e_pt_start: a 1-d numpy array listing x-y coordinate tuple of the start point of each undirected edge
    :param na_e_pt_end: a 1-d numpy array listing x-y coordinate tuple of the end point of each undirected edge
    :param file_dir: a string represents the file directory in which to store the exported drawing
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


def write_d_reach_to_table(tup_max_dc, tup_ang_threshold, input_dir, output_dir, filename, write_ddl=False):
    """Conduct directional reach analysis for the map stored in a CSV file.

    :param tup_max_dc: a tuple of the maximum number of direction changes allowed
    :param tup_ang_threshold: a tuple of the angle threshold used to determine whether to count as a direction change
    :param input_dir: the directory where the CSV file (to be read) is stored
    :param output_dir: the directory where the CSV file (with the results) is stored
    :param filename: the name (including the filename extension) of the original CSV file to be read and analyzed
    :param write_ddl: a boolean indicating whether to write DDL values to table
    """
    os.chdir(input_dir)
    df = pd.read_csv(filename)
    na_x1, na_y1, na_x2, na_y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_seg_len = df['seg_len'].values
    na_e_pt_id_matrix, na_e_pt_start, na_e_pt_end = get_directed_EV_id_matrix(na_x1, na_y1, na_x2, na_y2)
    for ang_threshold in tup_ang_threshold:
        adj_list = create_directed_adj_list(na_e_pt_id_matrix, na_e_pt_start, na_e_pt_end, ang_threshold)
        if write_ddl:
            col_name = 'DDL{}d'.format(ang_threshold)
            list_ddl = get_DDL_all_pairs(adj_list, na_seg_len)
            df[col_name] = pd.Series(list_ddl).values
        for max_dc in tup_max_dc:
            col_name = 'dr{0}dc{1}d'.format(max_dc, ang_threshold)
            list_dr = d_reach_all_pairs(max_dc, adj_list, na_seg_len)
            df[col_name] = pd.Series(list_dr).values
    filename = filename[:-4] + '_DR' + '.csv'
    os.chdir(output_dir)
    df.to_csv(filename, index=False)


def write_d_reach_to_tables(tup_max_dc, tup_ang_threshold, input_dir, output_dir, write_ddl=False):
    """Conduct directional reach analysis for all the CSV files stored in the input directory and export the results to
    the output directory.

    :param tup_max_dc: a tuple of the maximum number of direction changes allowed
    :param tup_ang_threshold: a tuple of the angle threshold used to determine whether to count as a direction change
    :param input_dir: the directory where the CSV file (to be read) is stored
    :param output_dir: the directory where the CSV file (with the results) is stored
    :param write_ddl: a boolean indicating whether to write DDL values to table
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            write_d_reach_to_table(tup_max_dc, tup_ang_threshold, input_dir, output_dir, filename,
                                   write_ddl)


if __name__ == '__main__':
    # Document the time at which the script starts running
    localtime = time.asctime(time.localtime(time.time()))
    print "Start Time :", localtime + "\n"

    # Change working directory
    directory = r"C:\_SoftwareDevelopment\Grasshopper\GhPython_PatternGeneration\data_RealExamples\Apt"
    os.chdir(directory)
    csv_file = "test_Apt.csv"
    # Set the threshold angle
    threshold_angle = 30
    # Set the number of direction changes allowed
    dc = 2
    # Set the start edge ID
    start_edge = 125

    # Read in the data and store them as a pandas DataFrame
    df = pd.read_csv(csv_file)

    x1, y1, x2, y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_segment_length = df['seg_len'].values

    na_edge_pt_id_matrix, na_edge_pt_start, na_edge_pt_end = get_directed_EV_id_matrix(x1, y1, x2, y2)
    # print na_edge_pt_id_matrix
    # print na_edge_pt_start
    # print na_edge_pt_end
    adjacency_list = create_directed_adj_list(na_edge_pt_id_matrix, na_edge_pt_start, na_edge_pt_end, threshold_angle)
    # print adjacency_list
    dr, reached_edges = d_reach(start_edge, dc, adjacency_list, na_segment_length)
    print "<seg. no.{}> dr-{}-dc-{}d: {} m".format(start_edge, dc, threshold_angle, dr)
    ddl = get_DDL(start_edge, adjacency_list, na_segment_length)
    print "<seg. no.{}> DDL-{}d: {}".format(start_edge, threshold_angle, ddl)
    mean_ddl = get_mean_DDL(adjacency_list, na_segment_length)
    print "Mean DDL: {}".format(mean_ddl)

    # Document the time at which the script finishes running
    localtime = time.asctime(time.localtime(time.time()))
    print "\nEnd Time :", localtime

    # Change the directory to save the PDF file
    directory = "C:/Users/cfeng/Desktop/Outbox"

    # Highlight the reached edges on the initial map and export the drawing to a PDF file
    draw_d_reach_edges(start_edge, reached_edges, na_edge_pt_start, na_edge_pt_end, directory, stroke_width=0.02,
                       scale_factor=1 / 50)
