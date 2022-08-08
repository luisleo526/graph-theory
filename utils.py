from multiprocessing import Process, Manager


def parallel_loop_task(f, n, cores, i, return_dict):
    start = i * int(n / cores)
    end = min(n, (i + 1) * int(n / cores))
    for j in range(start, end):
        result = f(j)
        return_dict[j] = result


def parallel_loop(f, n, max_cores):
    cores = min(max_cores, n)
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for p in range(cores):
        jobs.append(Process(target=parallel_loop_task, args=(f, n, cores, p, return_dict,)))

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    return list(return_dict.values())


def compute_X(edge, base=100):
    a, b = edge
    assert a <= b
    if a != b:
        return base * a + b
    else:
        return 0


def compute_Y(src1, src2, tgt1, tgt2):
    y1 = (compute_X(tgt1) - compute_X(tgt2))
    y2 = (compute_X(src1) - compute_X(src2))

    return y1 / y2


def compute_Z(src, tgt):
    assert len(src) == len(tgt)
    num_edges = len(src)
    result = 1
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            result *= compute_Y(src[i], src[j], tgt[i], tgt[j])

    return result


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
    result = 1
    for i in range(num_edge):
        for j in range(i + 1, num_edge):
            result *= compute_Y(graph[i], graph[j], h(graph[i], edge), h(graph[j], edge))

    return result


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
