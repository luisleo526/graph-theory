import numpy as np
import math
from sympy.matrices import Matrix
from sympy import symbols, LC, LM, poly
from sympy.utilities.iterables import multiset_permutations
from collections import Iterable
from time import process_time


def sign(x):
    return math.copysign(1, x)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def graph_permuation(src_graph, tgt_index, sort=False):
    tgt_index = list(flatten(tgt_index))
    src_index = [i + 1 for i in range(len(tgt_index))]
    mapping = dict(zip(tgt_index, src_index))

    tgt_graph = []
    for a, b in src_graph:
        c, d = mapping[a], mapping[b]
        if c > d:
            tgt_graph.append((d, c))
        else:
            tgt_graph.append((c, d))

    if sort:
        return sorted(tgt_graph)
    else:
        return tgt_graph


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
                result *= sign(compute_Y(graph_a[i], graph_a[j], graph_b[i], graph_b[j]))

    return result


class Graph:

    def __init__(self, graph):

        self.orientable = None
        self.equivalent = None
        self.z = None
        self.invar_poly = None
        self.permutation_sets = None
        self.permute_index = None
        self.invar_diffs = None
        self.invar_coeff = None
        self.invar = None
        self.adj = None
        self.graph = graph
        self.n = max([max(x) for x in self.graph])

    def calculate_invariant(self):

        self.adj = np.zeros((self.n, self.n), dtype=np.int8)
        for (i, j) in self.graph:
            self.adj[i - 1, j - 1] = 1
        self.adj += self.adj.transpose()

        self.invar_poly = []
        self.invar_coeff = []

        self.invar_poly.append(Matrix(self.adj).charpoly(symbols('x')).as_expr())
        self.invar_coeff.append(poly(self.invar_poly[-1]).all_coeffs())

        for i in range(self.n):
            _adj = np.delete(np.delete(self.adj, i, 0), i, 1)
            self.invar_poly.append(Matrix(_adj).charpoly(symbols('x')).as_expr())
            self.invar_coeff.append(poly(self.invar_poly[-1]).all_coeffs())
        self.invar = sorted(set([tuple(x) for x in self.invar_coeff]))

        # Compare polynomials
        self.invar_diffs = []
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
            self.invar_diffs.append(sum([1 for x in ans if x > 0]))

    def calculate_permutations(self):

        if self.invar_diffs is None:
            self.calculate_invariant()

        self.permute_index = []
        for e in sorted(set(self.invar_diffs)):
            vertices = []
            for i in range(len(self.invar_diffs)):
                if self.invar_diffs[i] == e:
                    vertices.append(i)
            self.permute_index.append(vertices)

        self.permute_index = self.permute_index[:-1]

        self.permutation_sets = []
        base = 1
        for index_set in self.permute_index:
            self.permutation_sets.append(list(multiset_permutations(index_set)))

    def print_info(self):

        if self.invar is None:
            self.calculate_invariant()

        if self.permute_index is None:
            self.calculate_permutations()

        if self.orientable is None:
            self.get_all_Z()

        print("graph:", self.graph)
        print("orientable:", self.orientable)
        print("permute index:", self.permute_index)
        print("invariant:")
        for j, invar in enumerate(self.invar):
            print(f"({j + 1}):", invar)

    def equal(self, graph):
        if self.invar is None:
            self.calculate_invariant()
        if graph.invar is None:
            graph.calculate_invariant()

        return set(self.invar) == set(graph.invar)

    def get_all_Z(self):

        if self.permutation_sets is None:
            self.calculate_permutations()

        if len(self.permutation_sets) == 1:
            return

        self.z = np.zeros([len(x) for x in self.permutation_sets], dtype=np.int8)
        self.equivalent = np.zeros([len(x) for x in self.permutation_sets], dtype=np.bool)

        it = np.nditer(self.z, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            permutation = []
            for i in range(len(self.permutation_sets)):
                permutation += self.permutation_sets[i][it.multi_index[i]]
            sub_graph = Graph(graph_permuation(self.graph, permutation))
            self.z[it.multi_index] = compute_Z(self.graph, sub_graph.graph)
            self.equivalent[it.multi_index] = self.equal(sub_graph)
            it.iternext()

        self.orientable = not np.any(-self.z * self.equivalent)


class AGraph(Graph):

    def __init__(self, graph):
        super(AGraph, self).__init__(graph)
        self.orientable = None

        self.calculate_invariant()
        self.calculate_permutations()
        self.standard_G = Graph(graph_permuation(src_graph=self.graph, tgt_index=self.permute_index, sort=True))

    def print_infos(self):
        self.standard_G.get_all_Z()
        self.get_all_Z()
        print("Original graph info:")
        print('-'*30)
        self.print_info()
        print("\nStandard graph info:")
        print('-' * 30)
        self.standard_G.print_info()


class GraphSets:

    def __init__(self, n=3):

        with open(f"./inputs/{2 * n:02d}_3_3.asc") as f:
            lines = f.readlines()

        lines = list(filter(''.__ne__, [x.strip() for x in lines]))

        start_index = [i for i, s in enumerate(lines) if 'Graph' in s]
        end_index = [i for i, s in enumerate(lines) if 'Taillenweite' in s]

        _graphs = []
        for start, end in zip(start_index, end_index):
            _graphs.append(lines[start + 1:end - 1])

        it_start = process_time()
        self.graphs = []
        for _graph in _graphs:
            graph = []
            for edge in _graph:
                vertices = edge.replace(' :', '').split(' ')
                for i in range(1, len(vertices)):
                    if int(vertices[i]) > int(vertices[0]):
                        graph.append((int(vertices[0]), int(vertices[i])))
            self.graphs.append(AGraph(graph=graph))
        it_end = process_time()
        print("CPU time:", it_end - it_start)

    def get_graph_info(self, i):
        self.graphs[i].print_infos()

    def number_of_graphs(self):
        return len(self.graphs)
