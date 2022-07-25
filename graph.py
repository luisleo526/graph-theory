import numpy as np
from munch import Munch
from sympy.matrices import Matrix
from sympy import symbols, LC, LM, poly
from sympy.utilities.iterables import multiset_permutations
from collections import Iterable
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import io
import networkx as nx
from math import isclose
import matplotlib.pyplot as plt
import math
import functools


def parallel_loop_task(f, n, cores, i, return_dict):
    start = i * int(n / cores)
    end = min(n, (i + 1) * int(n / cores))
    for j in range(start, end):
        result = f(j)
        return_dict[j] = result


def parallel_loop(f, n, max_cores):
    # with mp.Pool(processes=cores) as pool:
    #     results = pool.map(f, range(n))

    cores = max_cores
    if max_cores > n:
        cores = n

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


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


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

    # if isclose(result, int(result)):
    #     result = int(result)

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


def compute_Zr(graph, edge):
    assert edge in graph
    num_edge = len(graph)
    result = 1
    for i in range(num_edge):
        for j in range(i + 1, num_edge):
            result *= compute_Y(graph[i], graph[j], h(graph[i], edge), h(graph[j], edge))

    # if isclose(result, int(result)):
    #     result = int(result)

    return result


class Graph:
    """
         self << f  : f(G)
         abs(self)  : sG
         bool(self) : self.orientable
    """

    def __init__(self, graph, threads=1, src_graph=None, reduced_edge=None):

        self._orientable = None
        self._f_repr = None
        self._z_src = None
        self._z_sg = None
        self._z_repr = None
        self._sG = None
        self._er_sets = None
        self._permutation_sets = None
        self._permute_indices = None
        self._invar = None
        self._invar_poly = None
        self._adj = None

        self.name = None
        self.is_repr = None
        self.repr = None
        self.src_graph = src_graph
        self.reduced_edge = reduced_edge

        self.graph = graph
        self.order()
        self.threads = threads

    def __hash__(self):
        msg = ""
        for a, b in self.sort:
            msg += f"{a}{b}"
        return hash(int(msg))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __gt__(self, other):
        return hash(self) > hash(other)

    def __le__(self, other):
        return hash(self) <= hash(other)

    def __ge__(self, other):
        return hash(self) >= hash(other)

    def __abs__(self):
        return self.sG

    def __len__(self):
        return len(self.G)

    def __lshift__(self, f):

        tgt_index = list(flatten(f))
        src_index = list(flatten(self.permute_indices))
        mapping = dict(zip(src_index, tgt_index))

        tgt_graph = []
        for a, b in self.graph:
            c, d = mapping[a], mapping[b]
            if c > d:
                tgt_graph.append((d, c))
            else:
                tgt_graph.append((c, d))

        new_graph = Graph(tgt_graph, self.threads)

        return new_graph

    def order(self):
        graph = []
        for a, b in self.graph:
            if a > b:
                graph.append((b, a))
            else:
                graph.append((a, b))
        self.graph = graph

    @property
    def sort(self):
        return sorted(self.graph)

    @property
    def n(self):
        return max([max(x) for x in self.graph])

    @property
    def sG(self):
        if self._sG is None:
            self._sG = self << [x for x in range(1, self.n + 1)]
            self._sG.src_graph = self.src_graph
            self._sG.reduced_edge = self.reduced_edge
        return self._sG

    @property
    def G(self):
        return self.graph

    @property
    def adj(self):
        if self._adj is None:
            self._adj = np.zeros((self.n, self.n), dtype=np.int8)
            for (i, j) in self.G:
                self._adj[i - 1, j - 1] += 1
            self._adj += self._adj.transpose()
        return self._adj

    @property
    def is_valid(self):
        no_loop = np.trace(self.adj) == 0
        enough_edges = all(sum(x) > 2 for x in self.adj)
        is_symmetric = np.equal(self.adj, self.adj.transpose()).all()
        no_deprecate = set(list(np.unique(self.adj))) == {0, 1}
        return no_loop and enough_edges and is_symmetric and no_deprecate

    @property
    def invar_poly(self):
        if self._invar_poly is None:
            self._invar_poly = []
            self._invar_poly.append(Matrix(self.adj).charpoly(symbols('x')).as_expr())
            for i in range(self.n):
                _adj = np.delete(np.delete(self.adj, i, 0), i, 1)
                self._invar_poly.append(Matrix(_adj).charpoly(symbols('x')).as_expr())
        return self._invar_poly

    @property
    def invar(self):
        if self._invar is None:
            invar_coeff = []
            for expr in self.invar_poly:
                invar_coeff.append(poly(expr).all_coeffs())
            self._invar = sorted(set([tuple(x) for x in invar_coeff]))

        return self._invar

    @property
    def permute_indices(self):
        if self._permute_indices is None:
            invar_diffs = []
            for i in range(len(self.invar_poly)):
                ans = []
                for j in range(len(self.invar_poly)):
                    diff = self.invar_poly[i] - self.invar_poly[j]
                    if len(diff.free_symbols) == 0:
                        ans.append(0)
                    else:
                        if LC(diff) > 0:
                            k = i
                        else:
                            k = j
                        if abs(self.invar_poly[k].coeff(LM(diff))) > 1e-10:
                            ans.append(LC(diff))
                        else:
                            ans.append(-LC(diff))
                invar_diffs.append(sum([1 for x in ans if x > 0]))

            self._permute_indices = []
            for e in sorted(set(invar_diffs)):
                vertices = []
                for i in range(len(invar_diffs)):
                    if invar_diffs[i] == e:
                        vertices.append(i)
                self._permute_indices.append(vertices)

            self._permute_indices = self._permute_indices[:-1]

        return self._permute_indices

    @property
    def permutation_sets(self):
        if self._permutation_sets is None:
            self._permutation_sets = []
            for index_set in self.permute_indices:
                self._permutation_sets.append(list(multiset_permutations(index_set)))
        return self._permutation_sets

    @property
    def permutation_dim(self):
        return [len(x) for x in self.permutation_sets]

    @property
    def z_src(self):
        if self._z_src is None:
            self._z_src = compute_Zr(self.src_graph.sort, self.reduced_edge)
        return self._z_src

    @property
    def z_repr(self):
        if self._z_repr is None:
            self._z_repr = self.find_repr_z()
            self.repr._permutation_sets = None
        return self._z_repr

    @property
    def f_repr(self):
        if self._f_repr is None:
            _ = self.z_repr
        return self._f_repr

    @property
    def z_sg(self):
        if self._z_sg is None:
            self._z_sg = compute_Z(self.G, abs(self).G)
        return self._z_sg

    @property
    def z_mult(self):
        sign = functools.partial(math.copysign, 1)
        if type(self.z_repr) == str:
            return 0
        else:
            return int(sign(self.z_src)) * int(sign(self.z_repr)) * int(sign(self.z_sg))

    def _check_orientable(self, i):

        permutation = []
        indices = np.unravel_index(i, self.permutation_dim)
        for j in range(len(self.permutation_sets)):
            permutation += self.permutation_sets[j][indices[j]]
        sub_graph = self << permutation

        return abs(compute_Z(self.G, sub_graph.G) + 1.0) < 1e-10 and sub_graph == self

    @property
    def orientable(self):
        # For any z = -1 and f(G) = G, G is called non-orientable
        # result = not np.any((abs(self.z + 1.0) < 1e-10) * self.equiv)
        if self._orientable is None:
            n = np.prod(self.permutation_dim)
            if n < math.factorial(int(self.n))/10:
                results = parallel_loop(self._check_orientable, n, self.threads)
                self._orientable = not any(results)
            self._permutation_sets = None
        return self._orientable

    @property
    def nx_graph(self):
        nx_graph = nx.from_numpy_matrix(self.adj, create_using=nx.MultiGraph, parallel_edges=False)
        nx_graph = nx.relabel_nodes(nx_graph, dict(zip([x for x in range(self.n)], [x + 1 for x in range(self.n)])))
        return nx_graph

    def plot(self):
        return nx.draw(self.nx_graph, with_labels=True, pos=nx.shell_layout(self.nx_graph),
                       font_color="whitesmoke", font_size=18, node_size=600)

    def _find_repr_z(self, i):

        permutation = []
        indices = np.unravel_index(i, self.repr.permutation_dim)
        for j in range(len(self.repr.permutation_sets)):
            permutation += self.repr.permutation_sets[j][indices[j]]
        sub_graph = self.repr << permutation

        if sub_graph == abs(self):
            return compute_Z(self.repr.G, sub_graph.G), list(flatten(permutation))

    def find_repr_z(self):

        n = np.prod(self.repr.permutation_dim)
        results = parallel_loop(self._find_repr_z, n, self.threads)

        ans = []
        for r in results:
            if r is not None:
                z, f = r
                ans.append({"f": f, "z": z})

        zs = [x["z"] for x in ans]
        self._f_repr = [x["f"] for x in ans]

        if len(ans) > 1:
            assert sum([abs(x) - abs(zs[0]) for x in zs]) < 1e-10
            if abs(max(zs) - min(zs)) > 1e-10:
                pm = u"\u00B1"
                return f"{pm}{max(zs)}"
            else:
                return zs[0]
        else:
            return zs[0]

    @property
    def info(self):

        msg = ""
        msg += print_to_string("graph:", self.sort)
        msg += print_to_string("orientable:", self.orientable)
        msg += print_to_string("permute index:", self.permute_indices)
        msg += print_to_string("number of permutations:", np.prod(self.permutation_dim))
        msg += print_to_string("invariant:")
        for j, invar in enumerate(self.invar):
            msg += print_to_string(f"({j + 1}):", invar)
        return msg

    @property
    def infos(self):
        msg = ""
        msg += print_to_string("Original graph info:")
        msg += print_to_string('-' * 30)
        msg += self.info
        msg += print_to_string("\nStandard graph info:")
        msg += print_to_string('-' * 30)
        msg += abs(self).info
        return msg

    @property
    def data(self):
        msg = ""
        if self.reduced_edge is not None:
            e = self.src_graph.sort.index(self.reduced_edge) + 1
            msg += print_to_string(f"Z({self.src_graph.name:>5s}, {e:2d}):", self.z_src)
        msg += print_to_string(f"Z(   G , S ):", self.z_sg)
        msg += print_to_string(f"Z( S(G),{self.repr.name:>3s}):", self.z_repr)
        return msg

    def _release_memory(self):

        self._sG = None
        self._er_sets = None
        self._permutation_sets = None
        self._permute_indices = None
        self._invar = None
        self._invar_poly = None
        self._adj = None

    def release_memory(self):
        self._release_memory()
        abs(self)._release_memory()

    @property
    def er_sets(self):
        if self._er_sets is None:
            self._er_sets = []
            for i in range(len(self)):
                new_graph = []
                for j in range(len(self)):
                    if j != i:
                        new_graph.append(h(self.sort[j], self.sort[i]))
                _graph = Graph(graph=new_graph, threads=self.threads, src_graph=self, reduced_edge=self.sort[i])
                if _graph.is_valid:
                    self._er_sets.append(_graph)

        return self._er_sets


class GraphGroup:

    def __init__(self):
        self.graphs = []

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def append(self, item):
        self.graphs.append(item)

    def plot(self):
        n = len(self.graphs)
        rows = (n // 3 + min(1, n % 3) + 1)
        fig = plt.figure(figsize=(15, 5 * rows))
        for gid, g in enumerate(self.graphs, 1):
            ax = plt.subplot2grid((rows, 3), (gid // 3 + min(1, gid % 3), (gid - 1) % 3))
            g.plot()


class GraphManager:

    def __init__(self, name='A'):
        self.graphs = GraphGroup()
        self._repr = None
        self._o = None
        self._no = None
        self.name = name
        self._cand = None

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def append(self, item):
        self.graphs.append(item)

    def plot(self):
        self.graphs.plot()

    @property
    def repr(self):
        if self._repr is None:
            self._repr = GraphGroup()
            invar_list = []
            for g in sorted(self.graphs):
                if g.invar not in invar_list:
                    invar_list.append(g.invar)
                    g.is_repr = abs(g).is_repr = True
                    g.repr = abs(g).repr = abs(g)
                    self._repr.append(abs(g))
                else:
                    g.is_repr = abs(g).is_repr = False
                    for rg in self._repr:
                        if g.invar == rg.invar:
                            g.repr = abs(g).repr = rg
                            break
        return self._repr

    @property
    def o(self):
        if self._o is None:
            self._o = GraphGroup()
            cnt = 0
            for g in self.repr:
                if g.orientable:
                    cnt += 1
                    g.name = f"{self.name}{cnt}"
                    self._o.append(g)
        return self._o

    @property
    def no(self):
        if self._no is None:
            self._no = GraphGroup()
            cnt = 0
            for g in self.repr:
                if not g.orientable:
                    cnt += 1
                    g.name = f"{self.name}N{cnt}"
                    self._no.append(g)
        return self._no

    @property
    def cand(self):
        if self._cand is None:
            self._cand = GraphGroup()
            for g in sorted([g for g in self.graphs if not g.is_repr]):
                self._cand.append(g)
        return self._cand

    def group(self):
        _ = self.o
        _ = self.no


class GraphSets:

    def __init__(self, n=3, threads=1):
        self.graphs = Munch()

        self.graphs.A = GraphManager()
        for graph in readGraph(n):
            self.graphs.A.append(Graph(graph=graph, threads=threads))
        for g in self.A:
            g._orientable = abs(g)._orientable = True
        self.A.group()
        print("A readed")

    def __getattr__(self, item):

        if len(item) == 1 and ord('A') <= ord(item) <= ord('Z'):
            while item not in list(self.graphs.keys()):
                self.deeper_search()
            return getattr(self.graphs, item)
        else:
            raise AttributeError(f"Attribute {item} not found.")

    def deeper_search(self):

        last_type = chr(max([ord(x) for x in list(self.graphs.keys())]))
        next_type = chr(ord(last_type) + 1)

        _graphs = []
        for g in getattr(getattr(self.graphs, last_type), "repr"):
            _graphs += g.er_sets

        setattr(self.graphs, next_type, GraphManager(name=next_type))
        for g in _graphs:
            getattr(self.graphs, next_type).append(g)

        getattr(self.graphs, next_type).group()
