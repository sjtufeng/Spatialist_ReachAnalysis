# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002

from priodict import priorityDictionary


def Dijkstra(G, start, end=None):
    """Find shortest paths from the  start vertex to all vertices nearer than or equal to the end.

    The input graph G is assumed to have the following representation:
    A vertex can be any object that can be used as an index into a dictionary.
    G is a dictionary, indexed by vertices. For any vertex v, G[v] is itself a dictionary, indexed by
    the neighbors of v. For any edge v->w, G[v][w] is the length of the edge.

    Of course, G and G[v] need not be actual Python dict objects, they can be any other type of object
    that obeys dict protocol, for instance one could use a wrapper in which vertices are URLs of web pages
    and a call to G[v] loads the web page and finds its outgoing links.

    The output is a pair (D, P) where D[v] is the distance from start to v and P[v] is the predecessor
    of v along the shortest path from s to v.

    Dijkstra's algorithm is only guaranteed to work correctly when all edge lengths are positive.
    This code does not verify this property for all edges (only the edges examined until the end vertex
    is reached), but will correctly compute shortest paths even for some graphs with negative edges, and
    will raise an exception if it discovers that a negative edge has caused it to make a mistake.
    """
    D = {}  # dictionary of real shortest distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()  # estimated distances from "unsettled" or "unprocessed" vertices
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        if v == end:
            break

        for w in G[v]:
            vw_dist = D[v] + G[v][w]
            if w in D:
                if vw_dist < D[w]:
                    raise ValueError("Dijkstra: found better path to already-processed vertex")
            elif w not in Q or vw_dist < Q[w]:
                Q[w] = vw_dist
                P[w] = v

    return D, P


def DijkstraWithinLimit(G, start, max_cost, end=None):
    """Conduct directional-reach analysis."""
    D = {}  # dictionary of real shortest distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()  # estimated distances from "unsettled" or "unprocessed" vertices
    Q[start] = 0

    for v in Q:
        # if the rest vertices are more than max_cost away, then there's no need to proceed
        if Q[v] > max_cost:
            break

        D[v] = Q[v]
        if v == end:
            break

        for w in G[v]:
            vw_dist = D[v] + G[v][w]
            if w in D:
                if vw_dist < D[w]:
                    raise ValueError("Dijkstra: found better path to already-processed vertex")
            elif w not in Q or vw_dist < Q[w]:
                Q[w] = vw_dist
                P[w] = v

    return D, P


def shortestPath(G, start, end):
    """Find a single shortest path from the given start vertex to the given end vertex.

    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along the shortest path.
    """

    D, P = Dijkstra(G, start, end)
    Path = []
    while 1:
        Path.append(end)
        if end == start:
            break
        end = P[end]
    Path.reverse()
    return Path


# # example, CLR p.528
# G = {'s': {'u': 10, 'x': 5},
#      'u': {'v': 1, 'x': 2},
#      'v': {'y': 4},
#      'x': {'u': 3, 'v': 9, 'y': 2},
#      'y': {'s': 7, 'v': 6}}
#
# print(Dijkstra(G, 's'))
# print(shortestPath(G, 's', 'v'))