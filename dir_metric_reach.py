"""Directional-metric reach analysis.

Author: Chen Feng
Last updated on Nov. 21, 2018
"""

from __future__ import division
import numpy as np
import pandas as pd
import os
import time
from pyx import *
import dir_reach
import metric_reach


def d_m_reach(edge_id, max_dc, ang_threshold, max_dist, na_x1, na_y1, na_x2, na_y2, na_seg_len):
    """Conduct directional-metric reach analysis.

    :param edge_id: the ID of the source edge
    :param max_dc: the maximum number of directional changes allowed
    :param ang_threshold: the angle threshold used to determine whether to count as a direction change
    :param max_dist: the maximum metric distance that one is allowed to travel
    :param na_x1: a 1-d numpy array listing the x-coordinate of the start point of the undirected edge/segment
    :param na_y1: a 1-d numpy array listing the y-coordinate of the start point of the undirected edge/segment
    :param na_x2: a 1-d numpy array listing the x-coordinate of the end point of the undirected edge/segment
    :param na_y2: a 1-d numpy array listing the y-coordinate of the end point of the undirected edge/segment
    :param na_seg_len: a list listing the length of each line segment
    :return: the total street length accessible within max_dc direction changes and max_dist
    """
    na_edge_pts, na_pts_start, na_pts_end = dir_reach.get_directed_EV_id_matrix(na_x1, na_y1, na_x2, na_y2)
    adj_list = dir_reach.create_directed_adj_list(na_edge_pts, na_pts_start, na_pts_end, ang_threshold)
    dr, list_dr_edges = dir_reach.d_reach(edge_id, max_dc, adj_list, na_seg_len)
    # Use only the reached edges for the metric reach analysis
    subset_na_edge_pts = na_edge_pts[np.array(list_dr_edges)]
    subset_na_seg_len = na_seg_len[np.array(list_dr_edges)]
    new_adj_list = metric_reach.create_adj_list(subset_na_edge_pts, subset_na_seg_len)
    # Since the source edge will have a new index in the list of reached edges after the directional reach anlaysis,
    # we need to update it here
    new_edge_id = list_dr_edges.index(edge_id)
    total_len = metric_reach.m_reach(new_edge_id, max_dist, new_adj_list, subset_na_edge_pts, subset_na_seg_len)

    return total_len


def d_m_reach_all_pairs(max_dc, ang_threshold, max_dist, na_x1, na_y1, na_x2, na_y2, na_seg_len):
    """Conduct directional-metric reach analysis for all lines.

    :param max_dc: the maximum number of directional changes allowed
    :param ang_threshold: the angle threshold used to determine whether to count as a direction change
    :param max_dist: the maximum metric distance that one is allowed to travel
    :param na_x1: a 1-d numpy array listing the x-coordinate of the start point of the undirected edge/segment
    :param na_y1: a 1-d numpy array listing the y-coordinate of the start point of the undirected edge/segment
    :param na_x2: a 1-d numpy array listing the x-coordinate of the end point of the undirected edge/segment
    :param na_y2: a 1-d numpy array listing the y-coordinate of the end point of the undirected edge/segment
    :param na_seg_len: a list listing the length of each line segment
    :return: a list listing the directional-metric reach for all line segment
    """
    list_reach = []
    num_lines = len(na_seg_len)
    for e in range(num_lines):
        reach = d_m_reach(e, max_dc, ang_threshold, max_dist, na_x1, na_y1, na_x2, na_y2, na_seg_len)
        list_reach.append(reach)

    return list_reach


def d_m_reach_output_reached_edges(edge_id, max_dc, ang_threshold, max_dist, na_x1, na_y1, na_x2, na_y2, na_seg_len):
    """Conduct directional-metric reach anlaysis and output the edges (including fractions of original edges) that
    have been reached.

    :param edge_id: the ID of the source edge
    :param max_dc: the maximum number of directional changes allowed
    :param ang_threshold: the angle threshold used to determine whether to count as a direction change
    :param max_dist: the maximum metric distance that one is allowed to travel
    :param na_x1: a 1-d numpy array listing the x-coordinate of the start point of the undirected edge/segment
    :param na_y1: a 1-d numpy array listing the y-coordinate of the start point of the undirected edge/segment
    :param na_x2: a 1-d numpy array listing the x-coordinate of the end point of the undirected edge/segment
    :param na_y2: a 1-d numpy array listing the y-coordinate of the end point of the undirected edge/segment
    :param na_seg_len: a list listing the length of each line segment
    :return: the updated ID of the starting edge, the total street length accessible within max_dc direction changes
            and max_dist, the lists of the line segments and fractions of line segments that can be reached,
            the subsets of the edge-point tables, and the edge-point tables for the original undirected graph
    """
    na_edge_pts, na_pts_start, na_pts_end = dir_reach.get_directed_EV_id_matrix(na_x1, na_y1, na_x2, na_y2)
    adj_list = dir_reach.create_directed_adj_list(na_edge_pts, na_pts_start, na_pts_end, ang_threshold)
    dr, list_dr_edges = dir_reach.d_reach(edge_id, max_dc, adj_list, na_seg_len)
    # Use only the reached edges for the metric reach analysis
    subset_na_edge_pts = na_edge_pts[np.array(list_dr_edges)]
    subset_na_seg_len = na_seg_len[np.array(list_dr_edges)]
    subset_na_pts_start = na_pts_start[np.array(list_dr_edges)]
    subset_na_pts_end = na_pts_end[np.array(list_dr_edges)]
    new_adj_list = metric_reach.create_adj_list(subset_na_edge_pts, subset_na_seg_len)
    # Since the source edge will have a new index in the list of reached edges after the directional reach anlaysis,
    # we need to update it here
    new_edge_id = list_dr_edges.index(edge_id)
    dmr, list_e_in, overflow_ratio1, overflow_ratio2, list_e_overflow1, list_e_overflow2, e_start_overflow = \
        metric_reach.m_reach_output_reached_edges(
            new_edge_id, max_dist, new_adj_list, subset_na_edge_pts, subset_na_seg_len)
    return (new_edge_id, dmr, list_e_in, overflow_ratio1, overflow_ratio2, list_e_overflow1, list_e_overflow2,
            e_start_overflow, subset_na_pts_start, subset_na_pts_end, subset_na_seg_len,
            na_pts_start[:na_pts_start.shape[0] // 2], na_pts_end[:na_pts_end.shape[0] // 2])


def draw_d_m_reach_edges(new_eid, max_dist, list_e_in, overflow_ratio1, overflow_ratio2,
                         list_e_overflow1, list_e_overflow2, e_start_overflow,
                         subset_na_pts_start, subset_na_pts_end, subset_na_seg_len,
                         na_edge_pt_start, na_edge_pt_end,
                         file_dir, file_name="dir_metric_reach_snapshot", stroke_width=0.02, scale_factor=1 / 50):
    """Visualize the result of directional-metric reach analysis.

    :param new_eid: the ID (updated during the directional-metric reach analysis) of the edge from which
                    the reach analysis starts
    :param max_dist: the maximum metric distance allowed to travel
    :param list_e_in: the list of edges that have been entirely reach
    :param overflow_ratio1: (list) the fraction of the edge that has been reached (from the start point of the edge)
    :param overflow_ratio2: (list) the fraction of the edge that has been reached (from the end point of the edge)
    :param list_e_overflow1: the list of edges that have been partially reached from their start points
    :param list_e_overflow2: the list of edges that have been partially reached from their end points
    :param e_start_overflow: a boolean indicates whether the starting edge has only been partially reached
    :param subset_na_pts_start: the subset of na_pts_start derived after the directional reach analysis
    :param subset_na_pts_end: the subset of na_points_end derived after the directional reach analysis
    :param subset_na_seg_len: the subset of na_segLen derived after the directional reach analysis
    :param na_edge_pt_start: a 1-d numpy array listing x-y coordinate tuple of the start point of each undirected edge
    :param na_edge_pt_end: a 1-d numpy array listing x-y coordinate tuple of the end point of each undirected edge
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
    circle_center_x = (subset_na_pts_start[new_eid][0] + subset_na_pts_end[new_eid][0]) / 2
    circle_center_y = (subset_na_pts_start[new_eid][1] + subset_na_pts_end[new_eid][1]) / 2

    if e_start_overflow:
        e_end_to_p2_x = circle_center_x + (max_dist / (subset_na_seg_len[new_eid] / 2)) * (
                subset_na_pts_end[new_eid][0] - circle_center_x)
        e_end_to_p2_y = circle_center_y + (max_dist / (subset_na_seg_len[new_eid] / 2)) * (
                subset_na_pts_end[new_eid][1] - circle_center_y)
        line = path.line(circle_center_x * scale_factor, circle_center_y * scale_factor, e_end_to_p2_x * scale_factor,
                         e_end_to_p2_y * scale_factor)
        c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
        e_end_to_p1_x = circle_center_x - (max_dist / (subset_na_seg_len[new_eid] / 2)) * (
                circle_center_x - subset_na_pts_start[new_eid][0])
        e_end_to_p1_y = circle_center_y - (max_dist / (subset_na_seg_len[new_eid] / 2)) * (
                circle_center_y - subset_na_pts_start[new_eid][1])
        line = path.line(circle_center_x * scale_factor, circle_center_y * scale_factor, e_end_to_p1_x * scale_factor,
                         e_end_to_p1_y * scale_factor)
        c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
    else:
        # Draw the reached edges in list_e_in
        for e in list_e_in:
            line = path.line(subset_na_pts_start[e][0] * scale_factor, subset_na_pts_start[e][1] * scale_factor,
                             subset_na_pts_end[e][0] * scale_factor, subset_na_pts_end[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])

        # Draw the partially reached edges
        for i in range(len(list_e_overflow1)):
            e = list_e_overflow1[i]
            e_start_x = subset_na_pts_start[e][0] + overflow_ratio1[i] * (
                    subset_na_pts_end[e][0] - subset_na_pts_start[e][0])
            e_start_y = subset_na_pts_start[e][1] + overflow_ratio1[i] * (
                    subset_na_pts_end[e][1] - subset_na_pts_start[e][1])
            line = path.line(e_start_x * scale_factor, e_start_y * scale_factor,
                             subset_na_pts_start[e][0] * scale_factor, subset_na_pts_start[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])
        for i in range(len(list_e_overflow2)):
            e = list_e_overflow2[i]
            e_start_x = subset_na_pts_end[e][0] - overflow_ratio2[i] * (
                    subset_na_pts_end[e][0] - subset_na_pts_start[e][0])
            e_start_y = subset_na_pts_end[e][1] - overflow_ratio2[i] * (
                    subset_na_pts_end[e][1] - subset_na_pts_start[e][1])
            line = path.line(e_start_x * scale_factor, e_start_y * scale_factor, subset_na_pts_end[e][0] * scale_factor,
                             subset_na_pts_end[e][1] * scale_factor)
            c.stroke(line, [style.linewidth(4 * stroke_width), color.rgb.red])

    # Draw a red circle at the midpoint of the edge from which to start
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
    # Set the threshold angle
    threshold_angle = 20
    # Set the number of direction changes allowed
    dc = 1
    # Set the maximum metric distance allowed to travel
    max_distance = 150
    # Set the start edge ID
    start_edge = 125

    # Read in the data and store them as a pandas DataFrame
    df = pd.read_csv(csv_file)

    x1, y1, x2, y2 = df['x1'].values, df['y1'].values, df['x2'].values, df['y2'].values
    na_segment_length = df['seg_len'].values

    print d_m_reach(start_edge, dc, threshold_angle, max_distance, x1, y1, x2, y2, na_segment_length)

    (new_start_eid, dir_metric_reach, list_edge_in, list_overflow_ratio1, list_overflow_ratio2, list_edge_overflow1,
     list_edge_overflow2, edge_start_overflow, subset_na_points_start, subset_na_points_end, subset_na_segment_length,
     na_points_start, na_points_end) = d_m_reach_output_reached_edges(start_edge, dc, threshold_angle, max_distance, x1,
                                                                      y1, x2, y2, na_segment_length)
    # print d_m_reach_all_pairs(dc, threshold_angle, max_distance, x1, y1, x2, y2, na_segment_length)

    # Document the time at which the script finishes running
    localtime = time.asctime(time.localtime(time.time()))
    print "\nEnd Time :", localtime

    directory = "C:/Users/cfeng/Desktop/Outbox"
    # Highlight the reached edges on the initial map and export the drawing to a PDF file
    draw_d_m_reach_edges(new_start_eid, max_distance, list_edge_in, list_overflow_ratio1, list_overflow_ratio2,
                         list_edge_overflow1,
                         list_edge_overflow2, edge_start_overflow, subset_na_points_start, subset_na_points_end,
                         subset_na_segment_length,
                         na_points_start, na_points_end, directory,
                         stroke_width=0.02, scale_factor=1 / 50)
