from multiprocessing import Process, Manager
import numpy as np
from numpy.linalg import matrix_rank
from datetime import datetime


def parallel_loop_task(f, n, cores, i, return_dict):
    start = i * int(n / cores)
    end = min(n, (i + 1) * int(n / cores))
    if i + 1 == cores:
        end = n
    return_dict[i] = [f(j) for j in range(start, end)]


def parallel_loop(f, n, max_cores):
    cores = min(max_cores, n)

    with Manager() as manager:

        return_dict = manager.dict()
        jobs = []
        for p in range(cores):
            jobs.append(Process(target=parallel_loop_task, args=(f, n, cores, p, return_dict,)))

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        results = []
        for _result in list(return_dict.values()):
            results.extend(_result)

    return results


def compute_X(edge, base=100):
    a, b = edge
    assert a <= b
    if a != b:
        return int(base * a + b)
    else:
        return 0


def compute_Y(src1, src2, tgt1, tgt2):
    y1 = (compute_X(tgt1) - compute_X(tgt2))
    y2 = (compute_X(src1) - compute_X(src2))

    return y1, y2


def compute_Z(src, tgt):
    assert len(src) == len(tgt)
    num_edges = len(src)
    numerator = 1
    denominator = 1
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            num, den = compute_Y(src[i], src[j], tgt[i], tgt[j])
            numerator *= int(num)
            denominator *= int(den)

    if abs(numerator) == abs(denominator):
        if numerator + denominator == 0:
            return -1
        else:
            return 1
    else:
        return float(numerator) / float(denominator)


def h(edge, redge):
    c, d = edge
    a, b = redge

    assert b > a and d > c

    if c == b:
        c = a
    elif c > b:
        c = c - 1

    if d == b:
        d = a
    elif d > b:
        d = d - 1

    if c > d:
        return d, c
    else:
        return c, d


def compute_Zh(graph, edge):
    assert edge in graph
    num_edge = len(graph)
    numerator = 1
    denominator = 1
    for i in range(num_edge):
        for j in range(i + 1, num_edge):
            num, den = compute_Y(graph[i], graph[j], h(graph[i], edge), h(graph[j], edge))
            numerator *= int(num)
            denominator *= int(den)

    if abs(numerator) == abs(denominator):
        if numerator + denominator == 0:
            return -1
        else:
            return 1
    else:
        return float(numerator) / float(denominator)


def readGraph(n):
    with open(f"./inputs/{2 * n:02d}_3_3.asc") as f:
        lines = f.readlines()

    lines = list(filter(''.__ne__, [x.strip() for x in lines]))

    start_index = [i for i, s in enumerate(lines) if 'Graph' in s]
    end_index = [i for i, s in enumerate(lines) if 'Taillenweite' in s]

    _graphs = []
    for start, end in zip(start_index, end_index):
        _graphs.append(lines[start + 1:end - 1])

    graphs = []
    for _graph in _graphs:
        graph = []
        for edge in _graph:
            vertices = edge.replace(' :', '').split(' ')
            for i in range(1, len(vertices)):
                if int(vertices[i]) > int(vertices[0]):
                    graph.append((int(vertices[0]), int(vertices[i])))
        graphs.append(graph)

    return graphs


def get_data_task(p, n_cores, tgt_graphs, m, n, num_edges, return_dict):
    data = np.zeros((m, n), dtype=np.int)
    data2 = np.zeros((m, n + 1), dtype=object)
    edges = np.empty(m, dtype=object)

    for i in range(edges.shape[0]):
        edges[i] = [0 for _ in range(num_edges)]

    for i in range(data2.shape[0]):
        for j in range(data2.shape[1]):
            data2[i, j] = ""

    start = p * int(len(tgt_graphs) / n_cores)
    end = min(len(tgt_graphs), (p + 1) * int(len(tgt_graphs) / n_cores))

    if p + 1 == n_cores:
        end = len(tgt_graphs)

    for i in range(start, end):
        g = tgt_graphs[i]
        data[g.src.id, g.repr.id] += g.Zall
        data2[g.src.id, g.repr.id] += f"#{g.edge_index + 1}:[{g.Zh},{g.Zs},{g.Zr}], "
        edges[g.src.id][g.edge_index] += 1

    return_dict[p] = (data, data2, edges)


def get_data(src_graphs, tgt_graphs, cores, skip_rank=False):
    data = np.zeros((len(src_graphs.o) + len(src_graphs.no),
                     len(tgt_graphs.o) + len(tgt_graphs.no)),
                    dtype=np.int)

    data2 = np.zeros((len(src_graphs.o) + len(src_graphs.no),
                      len(tgt_graphs.o) + len(tgt_graphs.no) + 1),
                     dtype=object)
    edges = np.empty(len(src_graphs.o) + len(src_graphs.no), dtype=object)

    for i in range(edges.shape[0]):
        edges[i] = [0 for _ in range(len(src_graphs[0].sG.edges))]

    for i in range(data2.shape[0]):
        for j in range(data2.shape[1]):
            data2[i, j] = ""

    print(f"{datetime.now()}, Constructing {src_graphs.name + tgt_graphs.name} full matrix of size "
          f"{len(src_graphs.o) + len(src_graphs.no)}x{len(tgt_graphs.o) + len(tgt_graphs.no)}")

    # # Sequential execution
    # for g in tgt_graphs:
    #     data[g.src.id, g.repr.id] += g.Zall
    #     data2[g.src.id, g.repr.id] += f"#{g.edge_index + 1}:[{g.Zh},{g.Zs},{g.Zr}], "
    #     edges[g.src.id][g.edge_index] += 1

    # Parallel execution
    with Manager() as manager:

        return_dict = manager.dict()
        jobs = []
        for p in range(cores):
            jobs.append(Process(target=get_data_task, args=(p, cores, tgt_graphs,
                                                            len(src_graphs.o) + len(src_graphs.no),
                                                            len(tgt_graphs.o) + len(tgt_graphs.no),
                                                            len(src_graphs[0].sG.edges), return_dict,)))

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for a, b, c in list(return_dict.values()):
            data += a
            data2 += b
            edges += c

    for i in range(edges.shape[0]):
        for j in range(len(edges[i])):
            if edges[i][j] == 0:
                data2[i, -1] += f"#{j + 1}, "

    half_data = data[:len(src_graphs.o), :len(tgt_graphs.o)]

    rows = []
    columns = []
    for d, g in [[columns, src_graphs], [rows, tgt_graphs]]:
        for pref, l in [[g.name, len(g.o)], [g.name + 'N', len(g.no)]]:
            for i in range(l):
                d.append(pref + str(i + 1))

    if not skip_rank:
        print(f"{datetime.now()}, Computing rank of {src_graphs.name + tgt_graphs.name} half matrix of size "
              f"{len(src_graphs.o)}x{len(tgt_graphs.o)}")

        if len(src_graphs.o) > 0 and len(tgt_graphs.o) > 0:
            rank = matrix_rank(half_data)
        else:
            rank = 0
        return rows, columns, data2, data, half_data, rank
    else:
        return rows, columns, data2, data, half_data, None
