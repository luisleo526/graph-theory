import itertools
import sys
from collections.abc import Iterable

import numpy as np
from sympy import symbols, LC, LM, Poly
from sympy.matrices import Matrix

from utils import hash_invar

primes = np.array(
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
     109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
     233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
     367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
     499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
     643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
     797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
     947, 953, 967, 971, 977, 983, 991, 997], dtype=np.intc)


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

        self._has_triangle = None

    def __hash__(self):
        if self.hash is None:
            val = 1
            for i, (a, b) in enumerate(sorted(self.edges)):
                val = val * primes[a + 25] - primes[len(primes) - b] * primes[i + 25]
            self.hash = int(val)
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

        # tgt_index = list(flatten(f))
        # src_index = list(flatten(self.stdf))
        # mapping = dict(zip(src_index, tgt_index))

        tgt_graph = []
        for a, b in self.edges:
            tgt_graph.append((f[a], f[b]))

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
    def has_triangle(self):
        if self._has_triangle is None:
            a3 = abs(Poly(self.invar_poly[0]).all_coeffs()[3])
            self._has_triangle = a3 % 2 == 0 and a3 != 0
        return self._has_triangle

    @property
    def invar(self):
        if self._invar is None:
            self._invar = hash_invar(self.adj)
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
        return map(
            lambda indices: {a: b for a, b in zip(list(flatten(self.stdf)), list(flatten(indices)))},
            itertools.product(*[itertools.permutations(x) for x in self.stdf]))
