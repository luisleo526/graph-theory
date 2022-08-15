from graph_child import GraphChild
from utils import *
import numpy as np
import functools
import math

_sign = functools.partial(math.copysign, 1)


def sign(x):
    return int(_sign(x))


class GraphParent:

    def __init__(self, edges, threads=4, src=None, src_edge=None, edge_index=None):
        self._er_sets = None
        self._Zall = None
        self._Zh = None
        self._Zr = None
        self._Zs = None
        self._orientable = None
        self.G = GraphChild(edges)
        self.G.edges = sorted(self.G.edges)

        self.sG = self.G << [x for x in range(1, self.G.n + 1)]
        self.sG.edges = sorted(self.sG.edges)

        self.src = src
        self.src_edge = src_edge
        self.edge_index = edge_index
        self.threads = threads

        self.is_repr = False
        self.repr = None
        self.name = None

    def __lt__(self, other):
        return hash(self.sG) < hash(other.sG)

    def __gt__(self, other):
        return hash(self.sG) > hash(other.sG)

    def __le__(self, other):
        return hash(self.sG) <= hash(other.sG)

    def __ge__(self, other):
        return hash(self.sG) >= hash(other.sG)

    def check_orientable(self, i):
        permutation = []
        indices = np.unravel_index(i, self.sG.permutation_dim)
        for j in range(len(self.sG.permutation_sets)):
            permutation += self.sG.permutation_sets[j][indices[j]]
        fG = self.sG << permutation

        return abs(compute_Z(self.sG.edges, fG.edges) + 1.0) < 1e-10 and fG == self.sG

    @property
    def orientable(self):
        if self._orientable is None:
            n = np.prod(self.sG.permutation_dim)
            # results = parallel_loop(self.check_orientable, n, self.threads)
            # self._orientable = not any(results)
            self._orientable = True
            for i in range(n):
                result_is_true = self.check_orientable(i)
                if result_is_true:
                    self._orientable = False
                    break
            self.sG._permutation_sets = None
        return self._orientable

    @property
    def Zs(self):
        if self._Zs is None:
            sG = self.G << [x for x in range(1, self.G.n + 1)]
            self._Zs = sign(compute_Z(self.G.edges, sG.edges))
        return sign(self._Zs)

    def find_Zr(self, i):
        permutation = []
        indices = np.unravel_index(i, self.repr.sG.permutation_dim)
        for j in range(len(self.repr.sG.permutation_sets)):
            permutation += self.repr.sG.permutation_sets[j][indices[j]]
        fG = self.repr.sG << permutation

        if fG == self.sG:
            return compute_Z(self.repr.sG.edges, fG.edges)
        else:
            return None

    @property
    def Zr(self):
        if self._Zr is None:
            n = np.prod(self.repr.sG.permutation_dim)
            results = parallel_loop(self.find_Zr, n, self.threads)
            zs = [x for x in results if x is not None]
            assert len(zs) > 0
            assert sum([(abs(x) - abs(zs[0])) / abs(x) for x in zs]) < 1e-10
            if abs(max(zs) - min(zs)) / abs(max(zs)) > 1e-10:
                pm = u"\u00B1"
                # self._Zr = f"{pm}{abs(max(zs))}"
                self._Zr = 0
            else:
                self._Zr = sign(zs[0])
        return self._Zr

    @property
    def Zh(self):
        if self._Zh is None:
            self._Zh = sign(compute_Zh(self.src.sG, self.src_edge))
        return self._Zh

    @property
    def Zall(self):
        if self._Zall is None:
            self._Zall = self.Zh * self.Zr * self.Zs
        return self._Zall

    @property
    def er_sets(self):
        if self._er_sets is None:
            self._er_sets = []
            for i in range(len(self.sG)):
                new_graph = []
                for j in range(len(self.sG)):
                    if j != i:
                        new_graph.append(h(self.sG.edges[j], self.sG.edges[i]))
                _graph = GraphParent(edges=new_graph, threads=self.threads,
                                     src=self, src_edge=self.sG.edges[i], edge_index=i)
                if _graph.G.is_valid:
                    self._er_sets.append(_graph)
        return self._er_sets
