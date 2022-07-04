import numpy as np
from sympy.matrices import Matrix
from sympy import symbols, LC, LM, poly
from sympy.utilities.iterables import multiset_permutations
from collections import Iterable
import multiprocessing as mp
import io
import networkx as nx


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


def compute_Y(edge_a1, edge_a2, edge_b1, edge_b2):
    a, b = edge_a1
    c, d = edge_a2
    fa, fb = edge_b1
    fc, fd = edge_b2

    return ((100 * fa + fb) - (100 * fc + fd)) / ((100 * a + b) - (100 * c + d))


def compute_Z(graph_a, graph_b):
    num_edges = len(graph_a)
    result = 1
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            if i != j:
                result *= compute_Y(graph_a[i], graph_a[j], graph_b[i], graph_b[j])

    return result


class Graph:
    """
         self << f  : f(G)
         abs(self)  : sG
         bool(self) : self.orientable
    """

    def __init__(self, graph, threads=1):

        self._sG = None
        self._er_sets = None
        self._equiv = None
        self._z = None
        self._permutation_sets = None
        self._permute_indices = None
        self._invar = None
        self._invar_poly = None
        self._adj = None

        self.graph = graph
        self.sort()
        self.n = max([max(x) for x in self.graph])
        self.threads = threads

    def __hash__(self):
        msg = ""
        for a, b in self.G:
            msg += f"{a}{b}"
        return hash(msg)

    def __eq__(self, other):
        return np.array_equal(self.adj, other.adj)

    def __abs__(self):
        return self.sG

    def __lshift__(self, f):

        tgt_index = list(flatten(f))
        src_index = [i + 1 for i in range(self.n)]
        mapping = dict(zip(tgt_index, src_index))

        tgt_graph = []
        for a, b in self.graph:
            c, d = mapping[a], mapping[b]
            if c > d:
                tgt_graph.append((d, c))
            else:
                tgt_graph.append((c, d))

        new_graph = Graph(tgt_graph, self.threads)
        new_graph.sort()

        return new_graph

    def order(self):
        graph = []
        for a, b in self.graph:
            if a > b:
                graph.append((b, a))
            else:
                graph.append((a, b))
        self.graph = graph

    def sort(self):
        self.order()
        self.graph = sorted(self.graph)

    @property
    def sG(self):
        if self._sG is None:
            self._sG = self << self.permute_indices
        return self._sG

    @property
    def G(self):
        return self.graph

    @property
    def adj(self):
        if self._adj is None:
            self._adj = np.zeros((self.n, self.n), dtype=np.int8)
            for (i, j) in self.G:
                self._adj[i - 1, j - 1] = 1
            self._adj += self._adj.transpose()

        return self._adj

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
    def z(self):
        if self._z is None:
            self.get_z()
        return self._z

    @property
    def equiv(self):
        if self._equiv is None:
            self.get_z()
        return self._equiv

    @property
    def orientable(self):
        return not np.any(abs(x + 1.0) < 1e-10 and y for x, y in zip(self.z, self.equiv))

    def plot(self):
        graph = nx.from_numpy_matrix(self.adj, create_using=nx.MultiGraph)
        return nx.draw(graph, with_labels=True, pos=nx.shell_layout(graph))

    def _get_z(self, i):

        permutation = []
        indices = np.unravel_index(i, self.permutation_dim)
        for j in range(len(self.permutation_sets)):
            permutation += self.permutation_sets[j][indices[j]]
        sub_graph = self << permutation

        return i, compute_Z(self.G, sub_graph.graph), self == sub_graph

    def get_z(self):

        if np.prod(self.permutation_dim) > 1000000:
            return

        self._z = np.zeros(self.permutation_dim, dtype=np.float32).flatten()
        self._equiv = np.zeros(self.permutation_dim, dtype=bool).flatten()

        with mp.Pool(processes=self.threads) as pool:
            results = pool.map(self._get_z, [x for x in range(np.prod(self.permutation_dim))])

        for i, z, equ in results:
            self._z[i] = z
            self._equiv[i] = equ

    @property
    def info(self):

        msg = ""
        msg += print_to_string("graph:", self.G)
        msg += print_to_string("orientable:", self.orientable)
        msg += print_to_string("permute index:", self.permute_indices)
        msg += print_to_string("number of permutations:", np.prod(self.permutation_dim))
        msg += print_to_string("invariant:")
        for j, invar in enumerate(self.invar):
            msg += print_to_string(f"({j + 1}):", invar)

        return msg

    def release_memory(self):

        self._equiv = None
        self._z = None
        self._permutation_sets = None
        self._permute_indices = None
        self._invar = None
        self._invar_poly = None
        self._adj = None

    @property
    def er_sets(self):
        if self._er_sets is None:
            self._er_sets = []
            for i in range(self.adj.shape[0]):
                for j in range(i, self.adj.shape[0]):
                    if self.adj[i][j] == 1:
                        if all(x < 2 for x in self.adj[i] + self.adj[j]):
                            tri = np.triu(self.adj, 0)
                            tri[i] = tri[i] + tri[j]
                            _adj = np.delete(np.delete(tri, j, 0), j, 1)
                            _adj = _adj + _adj.transpose()
                            graphs = []
                            for ii in range(_adj.shape[0]):
                                for jj in range(ii, _adj.shape[0]):
                                    if _adj[ii, jj] == 1:
                                        graphs.append((ii + 1, jj + 1))
                            self._er_sets.append(graphs)
        return self._er_sets


class OGraph(Graph):

    def __init__(self, graph, threads=1):
        super(OGraph, self).__init__(graph, threads)
        self.sgraph = None

    def infos(self):
        msg = ""
        msg += print_to_string("Original graph info:")
        msg += print_to_string('-' * 30)
        msg += self.info
        msg += print_to_string("\nStandard graph info:")
        msg += print_to_string('-' * 30)
        msg += abs(self).info

        return msg

    def release_all_memory(self):
        self.release_memory()
        abs(self).release_memory()


class GraphSets:

    def __init__(self, n=3, threads=1):

        with open(f"./inputs/{2 * n:02d}_3_3.asc") as f:
            lines = f.readlines()

        lines = list(filter(''.__ne__, [x.strip() for x in lines]))

        start_index = [i for i, s in enumerate(lines) if 'Graph' in s]
        end_index = [i for i, s in enumerate(lines) if 'Taillenweite' in s]

        _graphs = []
        for start, end in zip(start_index, end_index):
            _graphs.append(lines[start + 1:end - 1])

        self.graphs = []
        for _graph in _graphs:
            graph = []
            for edge in _graph:
                vertices = edge.replace(' :', '').split(' ')
                for i in range(1, len(vertices)):
                    if int(vertices[i]) > int(vertices[0]):
                        graph.append((int(vertices[0]), int(vertices[i])))
            self.graphs.append(OGraph(graph=graph, threads=threads))

    def print_graph_info(self, i):
        print(self.graphs[i].infos())

    def number_of_graphs(self):
        return len(self.graphs)
