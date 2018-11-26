"""Metric reach analysis.

Authors: Chen Feng, Wenwen Zhang
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


# __________Create adjacency list__________

def create_adj_list(na_edge_pts, na_seg_len):
    """Create an adjacency list for the undirected graph generated from the initial map.

    :param na_edge_pts: a 2-d numpy array listing the start and end point IDs of each edge
    :param na_seg_len: a list showing the length of each line segment
    :return: an adjacency list, e.g., G = {0: {104: 34.0, 132: 28.1, 133: 25.0}, 1: {136: 55.9, 15: 63.3}, ...}
            indicates that the point 0 is adjacent with points 104, 132, and 133, and the metric distances from
            those points are G[0][104] = 34.0, G[0][132] = 28.1, and G[0][133] = 25.0.
    """
    adj_list = {}
    set_pts = list(set(na_edge_pts[..., 0].tolist() + na_edge_pts[..., 1].tolist()))
    for i in set_pts:
        adj_list[i] = {}
        # check which edges have vertex i as the start point
        t1 = np.where(na_edge_pts[:, 0] == i)[0]
        # add the vertices adjacent from i and the associated costs to the adjacency list
        for j in t1:
            v = na_edge_pts[j, 1]
            v_dist = na_seg_len[j]
            adj_list[i][v] = v_dist
        # check which edges have vertex i as the end point
        t2 = np.where(na_edge_pts[:, 1] == i)[0]
        # add the vertices adjacent to i and the associated costs to the adjacency list
        for j in t2:
            v = na_edge_pts[j, 0]
            v_dist = na_seg_len[j]
            adj_list[i][v] = v_dist

    return adj_list


# __________Run metric-reach analysis__________

def m_reach(edge_id, max_dist, adj_list, na_edge_pts, na_seg_len):
    """Conduct metric reach analysis.

    :param edge_id: the ID of the source edge
    :param max_dist: the maximum metric distance that one is allowed to travel
    :param adj_list: an adjacency list, e.g., G = {0: {104: 34.0, 132: 28.1, 133: 25.0}, 1: {136: 55.9, 15: 63.3}, ...}
                     indicates that the point 0 is adjacent with points 104, 132, and 133, and the metric distances from
                     those points are G[0][104] = 34.0, G[0][132] = 28.1, and G[0][133] = 25.0.
    :param na_edge_pts: a 2-d numpy array listing the start and end point IDs of each edge
    :param na_seg_len: a list listing the length of each line segment
    :return: the total street length accessible within max_dist
    """

    # Add the starting point (i.e., the midpoint of the source edge) to the adjacency list
    v_start = -1  # we can just use -1 to specify the ID of the starting point
    v_start_p1 = na_edge_pts[edge_id, 0]
    v_start_p2 = na_edge_pts[edge_id, 1]
    v_start_dist = na_seg_len[edge_id] / 2
    v_start_adj_list = {v_start_p1: v_start_dist, v_start_p2: v_start_dist}
    # Update the original adjacency list
    adj_list[v_start] = v_start_adj_list
    adj_list[v_start_p1][v_start] = v_start_dist
    adj_list[v_start_p2][v_start] = v_start_dist
    if v_start_p2 in adj_list[v_start_p1]:
        del adj_list[v_start_p1][v_start_p2]
    if v_start_p1 in adj_list[v_start_p2]:
        del adj_list[v_start_p2][v_start_p1]
    # Apply Dijkstra's shortest-path algorithm
    D, P = Dijkstra(adj_list, v_start)
    # Restore the original adjacency list
    del adj_list[v_start]
    del adj_list[v_start_p1][v_start]
    del adj_list[v_start_p2][v_start]
    adj_list[v_start_p1][v_start_p2] = na_seg_len[edge_id]
    adj_list[v_start_p2][v_start_p1] = na_seg_len[edge_id]

    # find all the vertices that have been reached within the predefined distance threshold
    list_v_in = {k: v for k, v in D.items() if v <= max_dist}
    # find all the vertices that have not been reached within the predefined distance threshold
    # list_v_out = {k: v for k, v in D.items() if v > max_dist}

    # Clarify entirely reached and partially reached edges
    list_e_in = {}  # list including edges that have been entirely reached
    list_e_overflow = {}  # list including portions of edges that have only been partially reached

    num_edges = len(na_seg_len)
    # Proceed if the starting edge has been entirely reached
    if (max_dist * 2) >= na_seg_len[edge_id]:
        # Select edges that have been entirely reached
        for i in range(0, num_edges):
            e_len = na_seg_len[i]
            # Since we changed the adjacency list by adding one more vertex (the midpoint of the starting edge,
            # we need to be careful in dealing with the starting edge
            if i != edge_id:
                v_p1 = na_edge_pts[i, 0]
                # if the edge has been entirely reached, then its two endpoints must have also been reached
                if v_p1 in list_v_in:
                    v_p2 = na_edge_pts[i, 1]
                    if v_p2 in list_v_in:
                        dist_res1 = max_dist - D[v_p1]  # check how further one can go from v_p1
                        dist_res2 = max_dist - D[v_p2]  # check how further one can go from v_p2
                        dist_res = dist_res1 + dist_res2
                        if dist_res >= e_len:
                            list_e_in[i] = e_len
            else:
                list_e_in[edge_id] = e_len
        # Select edges that have been partially reached
        for i in range(0, num_edges):
            if i not in list_e_in:
                v_p1 = na_edge_pts[i, 0]
                v_p2 = na_edge_pts[i, 1]
                if (v_p1 in list_v_in) or (v_p2 in list_v_in):
                    # check if the start point of the edge has been reached
                    if v_p1 in list_v_in:
                        e_overflow = max_dist - D[v_p1]
                        eid_overflow = "%d_%d->%d" % (i, v_p1, v_p2)
                        list_e_overflow[eid_overflow] = e_overflow
                    # check if the end point of the edge has been reached
                    if v_p2 in list_v_in:
                        e_overflow = max_dist - D[v_p2]
                        eid_overflow = "%d_%d->%d" % (i, v_p2, v_p1)
                        list_e_overflow[eid_overflow] = e_overflow

    # Proceed if even the starting edge has only been partially reached
    else:
        eid_overflow1 = "%d_%d->%d" % (edge_id, v_start, v_start_p1)
        eid_overflow2 = "%d_%d->%d" % (edge_id, v_start, v_start_p2)
        dist_res = max_dist
        list_e_overflow[eid_overflow1] = dist_res
        list_e_overflow[eid_overflow2] = dist_res

    # Compute metric reach
    total_len = 0
    for k in list_e_in:
        total_len += list_e_in[k]
    for k in list_e_overflow:
        total_len += list_e_overflow[k]

    return total_len


def m_reach_output_reached_edges(edge_id, max_dist, adj_list, na_edge_pts, na_seg_len):
    """Conduct metric reach analysis and output the edges (including fractions of original edges) that have been reached.

    :param edge_id: the ID of the source edge
    :param max_dist: the maximum metric distance that one is allowed to travel
    :param adj_list: an adjacency list, e.g., G = {0: {104: 34.0, 132: 28.1, 133: 25.0}, 1: {136: 55.9, 15: 63.3}, ...}
                     indicates that the point 0 is adjacent with points 104, 132, and 133, and the metric distances from
                     those points are G[0][104] = 34.0, G[0][132] = 28.1, and G[0][133] = 25.0.
    :param na_edge_pts: a 2-d numpy array listing the start and end point IDs of each edge
    :param na_seg_len: a list showing the length of each line segment
    :return: the total street length accessible within max_dist and the lists of the line segments and fractions of
            line segments that can be reached
    """

    # Add the starting point (i.e., the midpoint of the source edge) to the adjacency list
    v_start = -1  # we can just use -1 to specify the ID of the starting point
    v_start_p1 = na_edge_pts[edge_id, 0]
    v_start_p2 = na_edge_pts[edge_id, 1]
    v_start_dist = na_seg_len[edge_id] / 2
    v_start_adj_list = {v_start_p1: v_start_dist, v_start_p2: v_start_dist}
    # Update the original adjacency list
    adj_list[v_start] = v_start_adj_list
    adj_list[v_start_p1][v_start] = v_start_dist
    adj_list[v_start_p2][v_start] = v_start_dist
    if v_start_p2 in adj_list[v_start_p1]:
        del adj_list[v_start_p1][v_start_p2]
    if v_start_p1 in adj_list[v_start_p2]:
        del adj_list[v_start_p2][v_start_p1]
    # Apply Dijkstra's shortest-path algorithm
    D, P = Dijkstra(adj_list, v_start)
    # Restore the original adjacency list
    del adj_list[v_start]
    del adj_list[v_start_p1][v_start]
    del adj_list[v_start_p2][v_start]
    adj_list[v_start_p1][v_start_p2] = na_seg_len[edge_id]
    adj_list[v_start_p2][v_start_p1] = na_seg_len[edge_id]

    # find all the vertices that have been reached within the predefined distance threshold
    list_v_in = {k: v for k, v in D.items() if v <= max_dist}
    # find all the vertices that have not been reached within the predefined distance threshold
    # list_v_out = {k: v for k, v in D.items() if v > max_dist}

    # Clarify entirely reached and partially reached edges
    list_e_in = {}  # list including edges that have been entirely reached
    list_e_overflow = {}  # list including portions of edges that have only been partially reached

    # ----------for visualization purposes----------
    overflow_ratio1 = []
    overflow_ratio2 = []
    list_e_overflow1 = []
    list_e_overflow2 = []
    e_start_overflow = False
    # ----------------------------------------------
    num_edges = len(na_seg_len)
    # Proceed if the starting edge has been entirely reached
    if (max_dist * 2) >= na_seg_len[edge_id]:
        # Select edges that have been entirely reached
        for i in range(0, num_edges):
            e_len = na_seg_len[i]
            # Since we changed the adjacency list by adding one more vertex (the midpoint of the starting edge,
            # we need to be careful in dealing with the starting edge
            if i != edge_id:
                v_p1 = na_edge_pts[i, 0]
                # if the edge has been entirely reached, then its two endpoints must have also been reached
                if v_p1 in list_v_in:
                    v_p2 = na_edge_pts[i, 1]
                    if v_p2 in list_v_in:
                        dist_res1 = max_dist - D[v_p1]  # check how further one can go from v_p1
                        dist_res2 = max_dist - D[v_p2]  # check how further one can go from v_p2
                        dist_res = dist_res1 + dist_res2
                        if dist_res >= e_len:
                            list_e_in[i] = e_len
            else:
                list_e_in[edge_id] = e_len
        # Select edges that have been partially reached
        for i in range(0, num_edges):
            if i not in list_e_in:
                v_p1 = na_edge_pts[i, 0]
                v_p2 = na_edge_pts[i, 1]
                if (v_p1 in list_v_in) or (v_p2 in list_v_in):
                    e_len = na_seg_len[i]
                    # check if the start point of the edge has been reached
                    if v_p1 in list_v_in:
                        e_overflow = max_dist - D[v_p1]
                        # ----------for visualization purposes----------
                        ratio = e_overflow / e_len
                        overflow_ratio1.append(ratio)
                        list_e_overflow1.append(i)
                        # ----------------------------------------------
                        eid_overflow = "%d_%d->%d" % (i, v_p1, v_p2)
                        list_e_overflow[eid_overflow] = e_overflow
                    # check if the end point of the edge has been reached
                    if v_p2 in list_v_in:
                        e_overflow = max_dist - D[v_p2]
                        # ----------for visualization purposes----------
                        ratio = e_overflow / e_len
                        overflow_ratio2.append(ratio)
                        list_e_overflow2.append(i)
                        # ----------------------------------------------
                        eid_overflow = "%d_%d->%d" % (i, v_p2, v_p1)
                        list_e_overflow[eid_overflow] = e_overflow

    # Proceed if even the starting edge has only been partially reached
    else:
        eid_overflow1 = "%d_%d->%d" % (edge_id, v_start, v_start_p1)
        eid_overflow2 = "%d_%d->%d" % (edge_id, v_start, v_start_p2)
        dist_res = max_dist
        list_e_overflow[eid_overflow1] = dist_res
        list_e_overflow[eid_overflow2] = dist_res
        # ----------for visualization purposes----------
        e_start_overflow = True
        # ----------------------------------------------
    # Compute metric reach
    total_len = 0
    for k in list_e_in:
        total_len += list_e_in[k]
    for k in list_e_overflow:
        total_len += list_e_overflow[k]

    return total_len, list_e_in, overflow_ratio1, overflow_ratio2, list_e_overflow1, list_e_overflow2, e_start_overflow


def m_reach_all_pairs(max_dist, adj_list, na_edge_pts, na_seg_len):
    """Conduct metric reach analysis for all lines.

    :param max_dist: the maximum metric distance that one is allowed to travel
    :param adj_list: an adjacency list, e.g., G = {0: {104: 34.0, 132: 28.1, 133: 25.0}, 1: {136: 55.9, 15: 63.3}, ...}
                     indicates that the point 0 is adjacent with points 104, 132, and 133, and the metric distances from
                     those points are G[0][104] = 34.0, G[0][132] = 28.1, and G[0][133] = 25.0.
    :param na_edge_pts: a 2-d numpy array listing the start and end point IDs of each edge
    :param na_seg_len: a list showing the length of each line segment
    :return: a list listing the metric reach for each line segment
    """
    list_reach = []
    num_lines = len(na_seg_len)
    for e in range(num_lines):
        reach = m_reach(e, max_dist, adj_list, na_edge_pts, na_seg_len)
        list_reach.append(reach)

    return list_reach


def write_m_reach_to_table(tup_max_dist, unit_of_length, input_dir, output_dir, filename):
    """Conduct metric reach analysis for the map stored in a CSV file.

    :param tup_max_dist: a tuple of the metric radius (in terms of network distance)
    :param unit_of_length: the unit of length used to express the distance (e.g., 'm', 'km', 'ft', 'mi')
    :param input_dir: the directory where the CSV file (to be read) is stored
    :param output_dir: the directory where the CSV file (with the results) is stored
    :param filename: the name (including the filename extension) of the original CSV file to be read and analyzed
    """
    os.chdir(input_dir)
    df = pd.read_csv(filename)
    na_x1, na_y1, na_x2, na_y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_seg_len = df['seg_len'].values
    na_edge_pts, na_pts_start, na_pts_end = get_EV_id_matrix(na_x1, na_y1, na_x2, na_y2)
    adj_list = create_adj_list(na_edge_pts, na_seg_len)
    for max_dist in tup_max_dist:
        col_name = 'mr{}{}'.format(max_dist, unit_of_length)
        list_mr = m_reach_all_pairs(max_dist, adj_list, na_edge_pts, na_seg_len)
        df[col_name] = pd.Series(list_mr).values
    filename = filename[:-4] + '_MR' + '.csv'
    os.chdir(output_dir)
    df.to_csv(filename, index=False)


def write_m_reach_to_tables(tup_max_dist, unit_of_length, input_dir, output_dir):
    """Conduct metric reach analysis for all the CSV files stored in the input directory and export the results to
        the output directory.

    :param tup_max_dist: a tuple of the metric radius (in terms of network distance)
    :param unit_of_length: the unit of length used to express the distance (e.g., 'm', 'km', 'ft', 'mi')
    :param input_dir: the directory where the CSV file (to be read) is stored
    :param output_dir: the directory where the CSV file (with the results) is stored
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            write_m_reach_to_table(tup_max_dist, unit_of_length, input_dir, output_dir, filename)


def draw_m_reach_edges(eid, max_dist, list_e_in, overflow_ratio1, overflow_ratio2, list_e_overflow1, list_e_overflow2,
                       e_start_overflow, na_edge_pt_start, na_edge_pt_end, na_seg_len, file_dir,
                       file_name="metric_reach_snapshot",
                       stroke_width=0.02, scale_factor=1 / 50):
    """visualize the result of metric reach analysis.

    :param eid: the ID of the edge from which the reach analysis starts
    :param max_dist: the maximum metric distance allowed to travel
    :param list_e_in: the list of edges that have been entirely reached
    :param overflow_ratio1: (list) the fraction of the edge that has been reached (from the start point of the edge)
    :param overflow_ratio2: (list) the fraction of the edge that has been reached (from the end point of the edge)
    :param list_e_overflow1: the list of edges that have been partially reached from their start points
    :param list_e_overflow2: the list of edges that have been partially reached from their end points
    :param e_start_overflow: a boolean indicates whether the starting edge has only been partially reached
    :param na_edge_pt_start: a 1-d numpy array listing x-y coordinate tuple of the start point of each undirected edge
    :param na_edge_pt_end: a 1-d numpy array listing x-y coordinate tuple of the end point of each undirected edge
    :param na_seg_len: a list showing the length of each line segment
    :param file_dir: a string represents the file directory in which to store the exported drawing
    :param file_name: the name of the PDF file to be exported to
    :param stroke_width: the width of the stroke for drawing lines
    :param scale_factor: the scale factor used to scale the input coordinates
    """
    os.chdir(file_dir)
    # Set canvas for drawing
    c = canvas.canvas()
    num_edges = len(na_edge_pt_start)

    # Draw the whole grid (i.e., the initial map) first
    for i in range(num_edges):
        line = path.line(na_edge_pt_start[i][0] * scale_factor, na_edge_pt_start[i][1] * scale_factor,
                         na_edge_pt_end[i][0] * scale_factor, na_edge_pt_end[i][1] * scale_factor)
        c.stroke(line, [style.linewidth(stroke_width), color.rgb.black])

    # Compute the coordinate of the midpoint of the starting edge
    circle_center_x = (na_edge_pt_start[eid][0] + na_edge_pt_end[eid][0]) / 2
    circle_center_y = (na_edge_pt_start[eid][1] + na_edge_pt_end[eid][1]) / 2

    if e_start_overflow:
        e_end_to_p2_x = \
            circle_center_x + (max_dist / (na_seg_len[eid] / 2)) * (na_edge_pt_end[eid][0] - circle_center_x)
        e_end_to_p2_y = \
            circle_center_y + (max_dist / (na_seg_len[eid] / 2)) * (na_edge_pt_end[eid][1] - circle_center_y)
        line = path.line(scale_factor * circle_center_x, scale_factor * circle_center_y, scale_factor * e_end_to_p2_x,
                         scale_factor * e_end_to_p2_y)
        c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
        e_end_to_p1_x = circle_center_x - (max_dist / (na_seg_len[eid] / 2)) * (
                circle_center_x - na_edge_pt_start[eid][0])
        e_end_to_p1_y = circle_center_y - (max_dist / (na_seg_len[eid] / 2)) * (
                circle_center_y - na_edge_pt_start[eid][1])
        line = path.line(scale_factor * circle_center_x, scale_factor * circle_center_y, scale_factor * e_end_to_p1_x,
                         scale_factor * e_end_to_p1_y)
        c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
    else:
        # Draw the reached edges in list_e_in
        for e in list_e_in:
            line = path.line(na_edge_pt_start[e][0] * scale_factor, na_edge_pt_start[e][1] * scale_factor,
                             na_edge_pt_end[e][0] * scale_factor, na_edge_pt_end[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])

        # Draw the partially reached edges
        for i in range(len(list_e_overflow1)):
            e = list_e_overflow1[i]
            e_start_x = na_edge_pt_start[e][0] + overflow_ratio1[i] * (na_edge_pt_end[e][0] - na_edge_pt_start[e][0])
            e_start_y = na_edge_pt_start[e][1] + overflow_ratio1[i] * (na_edge_pt_end[e][1] - na_edge_pt_start[e][1])
            line = path.line(scale_factor * e_start_x, scale_factor * e_start_y, na_edge_pt_start[e][0] * scale_factor,
                             na_edge_pt_start[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
        for i in range(len(list_e_overflow2)):
            e = list_e_overflow2[i]
            e_start_x = na_edge_pt_end[e][0] - overflow_ratio2[i] * (na_edge_pt_end[e][0] - na_edge_pt_start[e][0])
            e_start_y = na_edge_pt_end[e][1] - overflow_ratio2[i] * (na_edge_pt_end[e][1] - na_edge_pt_start[e][1])
            line = path.line(scale_factor * e_start_x, scale_factor * e_start_y, na_edge_pt_end[e][0] * scale_factor,
                             na_edge_pt_end[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])

    # Draw a red circle at the midpoint of the edge from which to start
    circle = path.circle(scale_factor * circle_center_x, scale_factor * circle_center_y, 8 * stroke_width)
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
    # Set the unit of length
    length_unit = 'm'
    # Set the maximum metric distance allowed to travel
    max_distance = 200
    # Set the start edge ID
    start_edge = 125

    # Read in the data and store them as a pandas DataFrame
    df = pd.read_csv(csv_file)

    x1, y1, x2, y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_segment_length = df['seg_len'].values

    na_edge_points, na_points_start, na_points_end = get_EV_id_matrix(x1, y1, x2, y2)
    adjacency_list = create_adj_list(na_edge_points, na_segment_length)
    mr = m_reach(start_edge, max_distance, adjacency_list, na_edge_points, na_segment_length)
    print "<seg. no.{}> mr-{}{}: {}{}".format(start_edge, max_distance, length_unit, mr, length_unit)

    (mr, list_edge_in, list_overflow_ratio1, list_overflow_ratio2, list_edge_overflow1, list_edge_overflow2,
     edge_start_overflow) = m_reach_output_reached_edges(start_edge, max_distance, adjacency_list, na_edge_points,
                                                         na_segment_length)

    # write_m_reach_to_table((50, 100, 200), length_unit, directory, directory, csv_file)

    # Document the time at which the script finishes running
    localtime = time.asctime(time.localtime(time.time()))
    print "\nEnd Time :", localtime

    # Change the directory to save the PDF file
    directory = "C:/Users/cfeng/Desktop/Outbox"
    # Highlight the reached edges on the initial map and export the drawing to a PDF file
    draw_m_reach_edges(start_edge, max_distance, list_edge_in, list_overflow_ratio1, list_overflow_ratio2,
                       list_edge_overflow1, list_edge_overflow2, edge_start_overflow, na_points_start, na_points_end,
                       na_segment_length, directory, stroke_width=0.02, scale_factor=1 / 50)
