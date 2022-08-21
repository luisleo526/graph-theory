import math
import sys
from collections.abc import Iterable

import numpy as np
from sympy import symbols, LC, LM
from sympy.matrices import Matrix
from sympy.utilities.iterables import multiset_permutations

from utils import find_adj_invar_coeffs


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class GraphChild:

    def __init__(self, edges):

        self.hash = None
        self._permutation_sets = None
        self._stdf = None
        self._invar = None
        self._invar_poly = None
        self._adj = None
        self._n = None

        self.edges = []
        for a, b in edges:
            if a > b:
                self.edges.append((b, a))
            else:
                self.edges.append((a, b))

    def __hash__(self):
        if self.hash is None:
            val = ""
            for a, b in sorted(self.edges):
                val += f"{a}{b}"
            self.hash = hash(int(val))
        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.adj, other.adj)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __gt__(self, other):
        return hash(self) > hash(other)

    def __le__(self, other):
        return hash(self) <= hash(other)

    def __ge__(self, other):
        return hash(self) >= hash(other)

    def __lshift__(self, f):

        tgt_index = list(flatten(f))
        src_index = list(flatten(self.stdf))
        mapping = dict(zip(src_index, tgt_index))

        tgt_graph = []
        for a, b in self.edges:
            tgt_graph.append((mapping[a], mapping[b]))

        new_graph = GraphChild(tgt_graph)

        return new_graph

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, item):
        return self.edges[item]

    @property
    def sort(self):
        return sorted(self.edges)

    @property
    def n(self):
        if self._n is None:
            self._n = max([max(x) for x in self.edges])
        return self._n

    @property
    def adj(self):
        if self._adj is None:
            self._adj = np.zeros((self.n, self.n), dtype=np.int8)
            for (i, j) in self.edges:
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
            self._invar = find_adj_invar_coeffs(self.adj)
        return self._invar

    @property
    def stdf(self):
        if self._stdf is None:
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
                        if abs(self.invar_poly[k].coeff(LM(diff))) > sys.float_info.epsilon * 2:
                            ans.append(LC(diff))
                        else:
                            ans.append(-LC(diff))
                invar_diffs.append(sum([1 for x in ans if x > 0]))

            self._stdf = []
            for e in sorted(set(invar_diffs)):
                vertices = []
                for i in range(len(invar_diffs)):
                    if invar_diffs[i] == e:
                        vertices.append(i)
                self._stdf.append(vertices)

            self._stdf = self._stdf[:-1]

        return self._stdf

    @property
    def permutation_sets(self):
        if self._permutation_sets is None:
            self._permutation_sets = []
            for index_set in self.stdf:
                self._permutation_sets.append(list(multiset_permutations(index_set)))
        return self._permutation_sets

    @property
    def permutation_dim(self):
        return [math.factorial(len(x)) for x in self.stdf]
