import functools
import math

from graph_child import GraphChild, flatten
from utils import *

_sign = functools.partial(math.copysign, 1)


def sign(x):
    return int(_sign(x))


class GraphParent:

    def __init__(self, edges, threads=4, src=None, src_edge=None, edge_index=None):
        self._unbind = None
        self.unbind_from_data = None
        self._er_sets = None
        self._Zall = None
        self._Zh = None
        self._Zr = None
        self._Zs = None
        self._orientable = None
        self.G = GraphChild(edges)
        self.G.edges = sorted(self.G.edges)

        self.sG = self.G << {y: x for x, y in enumerate(list(flatten(self.G.stdf)), 1)}
        self.sG.edges = sorted(self.sG.edges)

        self.src = src
        self.src_edge = src_edge
        self.edge_index = edge_index
        self.threads = threads

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

    def not_orientable(self, f):
        fG = self.sG << f
        return fG == self.sG and check_W(self.sG.edges, fG.edges) and compute_Z(self.sG.edges, fG.edges) == -1

    @property
    def orientable(self):
        if self._orientable is None:
            self._orientable = True
            for f in self.sG.permutation_sets:
                if self.not_orientable(f):
                    self._orientable = False
                    break
        return self._orientable

    @property
    def Zs(self):
        if self._Zs is None:
            sG = self.G << {y: x for x, y in enumerate(list(flatten(self.G.stdf)), 1)}
            self._Zs = sign(compute_Z(self.G.edges, sG.edges))
        return sign(self._Zs)

    def find_Zr(self, f):
        fG = self.repr.sG << f
        if fG == self.sG:
            return compute_Z(self.repr.sG.edges, fG.edges)
        else:
            return None

    @property
    def Zr(self):
        if self._Zr is None:
            zs = []
            for f in self.repr.sG.permutation_sets:
                z = self.find_Zr(f)
                if z is not None:
                    zs.append(z)
            assert len(zs) > 0
            assert sum([(abs(x) - abs(zs[0])) / abs(x) for x in zs]) < 1e-10
            if abs(max(zs) - min(zs)) / abs(max(zs)) > 1e-10:
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
            for i in range(len(self.sG.edges)):
                new_graph = []
                for j in range(len(self.sG.edges)):
                    if j != i:
                        new_graph.append(h(self.sG.edges[j], self.sG.edges[i]))
                _graph = GraphParent(edges=new_graph, threads=self.threads,
                                     src=self, src_edge=self.sG.edges[i], edge_index=i)
                if _graph.G.is_valid:
                    self._er_sets.append(_graph)
        return self._er_sets

    @property
    def unbind(self):
        if self._unbind is None:
            self._unbind = find_unbind_number(self.sG.adj)
        return self._unbind
