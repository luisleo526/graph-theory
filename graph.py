import numpy as np
from munch import Munch
from sympy.matrices import Matrix
from sympy import symbols, LC, LM, poly
from sympy.utilities.iterables import multiset_permutations
from collections import Iterable
import multiprocessing as mp
import io
import networkx as nx


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

    def __init__(self, graph, threads=1, src_graph=None, reduced_edge=None):

        self._sG = None
        self._er_sets = None
        self._equiv = None
        self._z = None
        self._permutation_sets = None
        self._permute_indices = None
        self._invar = None
        self._invar_poly = None
        self._adj = None

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
        return hash(msg)

    def __eq__(self, other):
        return np.array_equal(self.adj, other.adj)

    def __abs__(self):
        return self.sG

    def __len__(self):
        return len(self.G)

    def __lshift__(self, f):

        tgt_index = list(flatten(f))
        src_index = list(flatten(self.permute_indices))
        mapping = dict(zip(tgt_index, src_index))

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
            self._sG = self << self.permute_indices
            self._sG.graph = self._sG.sort
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
        # For any z = -1 and f(G) = G, G is called non-orientable
        result = not np.any((abs(self.z + 1.0) < 1e-10) * self.equiv)
        return result

    @property
    def nx_graph(self):
        nx_graph = nx.from_numpy_matrix(self.adj, create_using=nx.MultiGraph, parallel_edges=False)
        nx_graph = nx.relabel_nodes(nx_graph, dict(zip([x for x in range(self.n)], [x + 1 for x in range(self.n)])))
        return nx_graph

    def plot(self):
        return nx.draw(self.nx_graph, with_labels=True, pos=nx.shell_layout(self.nx_graph),
                       font_color="whitesmoke", font_size=18, node_size=600)

    def _get_z(self, i):

        permutation = []
        indices = np.unravel_index(i, self.permutation_dim)
        for j in range(len(self.permutation_sets)):
            permutation += self.permutation_sets[j][indices[j]]
        sub_graph = self << permutation

        return i, compute_Z(self.G, sub_graph.graph), self == sub_graph, sub_graph.sort, list(flatten(permutation))

    def get_z(self):

        # if np.prod(self.permutation_dim) > 1000000:
        #     return

        self._z = np.zeros(self.permutation_dim, dtype=np.float32).flatten()
        self._equiv = np.zeros(self.permutation_dim, dtype=bool).flatten()

        with mp.Pool(processes=self.threads) as pool:
            results = pool.map(self._get_z, [x for x in range(np.prod(self.permutation_dim))])

        for i, z, equ, _t0, _t1 in results:
            self._z[i] = z
            self._equiv[i] = equ

    def find_repr_z(self):

        with mp.Pool(processes=self.repr.threads) as pool:
            results = pool.map(self.repr._get_z, [x for x in range(np.prod(self.repr.permutation_dim))])

        ans = []
        for _t0, z, _t1, g, f in results:
            if g == self.sort:
                ans.append({"f": f, "z": z})
        return ans

    @property
    def info(self):

        msg = ""
        msg += print_to_string("graph:", self.G)
        msg += print_to_string("orientable:", self.orientable)
        msg += print_to_string("Z(G, sG):", compute_Z(self.graph, abs(self).graph))
        if self.is_repr is not None:
            msg += print_to_string("is representative:", self.is_repr)
            if not self.is_repr and self.repr is not None:
                msg += print_to_string("-" * 30)
                msg += print_to_string("rG:", self.repr.graph)
                msg += print_to_string("f, Z(G, rG):")
                for i, info in enumerate(self.find_repr_z(), 1):
                    msg += print_to_string(f"({i}): {str(info['f'])}, {info['z']:+.6f}")
                msg += print_to_string("-" * 30)
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

    def _release_memory(self):

        self._equiv = None
        self._z = None
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
                a, b = self.G[i]
                for j in range(len(self)):
                    if j != i:
                        c, d = self.G[j]
                        if c == a or c == b:
                            c = a
                        if d == a or d == b:
                            d = a

                        if c > b:
                            c -= 1

                        if d > b:
                            d -= 1
                        new_graph.append((c, d))
                _graph = Graph(graph=new_graph, threads=self.threads, src_graph=self, reduced_edge=(a, b))
                if _graph.is_valid:
                    self._er_sets.append(_graph)

        return self._er_sets


class GraphManager:

    def __init__(self):
        self.graphs = []
        self._repr = None
        self.o = []
        self.no = []

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def append(self, item):
        self.graphs.append(item)

    @property
    def repr(self):
        if self._repr is None:
            self._repr = []
            invar_list = []
            for g in self.graphs:
                if g.invar not in invar_list:
                    invar_list.append(g.invar)
                    self._repr.append(g)
                    g.is_repr = abs(g).is_repr = True
                else:
                    g.is_repr = abs(g).is_repr = False
                    for rg in self._repr:
                        if g.invar == rg.invar:
                            g.repr = abs(g).repr = rg
                            break
            for g in self._repr:
                if abs(g).orientable:
                    self.o.append(g)
                else:
                    self.no.append(g)

        return self._repr


class GraphSets:

    def __init__(self, n=3, threads=1):
        self.graphs = Munch()

        self.graphs.A = GraphManager()
        for graph in readGraph(n):
            self.graphs.A.append(Graph(graph=graph, threads=threads))
        _ = self.graphs.A.repr

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

        _graphs = set()
        for g in getattr(self.graphs, last_type):
            _graphs = _graphs.union(abs(g).er_sets)

        setattr(self.graphs, next_type, GraphManager())
        for g in _graphs:
            getattr(self.graphs, next_type).append(g)

        _ = getattr(self.graphs, next_type).repr
