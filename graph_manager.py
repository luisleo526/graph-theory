from graph_family import GraphFamily
from munch import Munch
from utils import readGraph
import numpy as np
from sympy.matrices import Matrix


class GraphManager:

    def __init__(self, n, threads):

        self.graphs = Munch()

        self.graphs.A = GraphFamily(readGraph(n), threads)
        for g in self.graphs.A:
            g._orientable = True
        self.graphs.A.set_repr()

        for i in range(26):
            last_type = chr(ord('A') + i)
            next_type = chr(ord('A') + i + 1)
            print(f"Finding {next_type} graphs...")
            next_graphs = getattr(self.graphs, last_type).deeper_graphs()
            if len(next_graphs) > 0:
                setattr(self.graphs, next_type, GraphFamily(next_graphs, threads, next_type))
                print(f"Finding representatives for {next_type} graphs...")
                getattr(self.graphs, next_type).set_repr()
            else:
                self.maxi = i
                break

    def __getattr__(self, item):

        if item in list(self.graphs.keys()):
            return getattr(self.graphs, item)
        else:
            raise AttributeError(f"Attribute {item} not found.")

    def get_matrix(self, src, tgt):
        src_graphs = getattr(self.graphs, src)
        tgt_graphs = getattr(self.graphs, tgt)
        data = np.zeros((len(src_graphs.o) + len(src_graphs.no),
                         len(tgt_graphs.o) + len(tgt_graphs.no)),
                        dtype=np.int)
        for g in tgt_graphs:
            data[g.src.id, g.repr.id] += g.Zall

        rows = []
        columns = []
        for d, g in [[columns, src_graphs], [rows, tgt_graphs]]:
            for pref, l in [[g.name, len(g.o)], [g.name + 'N', len(g.no)]]:
                for i in range(l):
                    d.append(pref + str(i + 1))

        return data, data[:len(src_graphs.o), :len(tgt_graphs.o)], rows, columns

    def get_all_data(self):
        data = Munch()
        for i in range(self.maxi):
            src = chr(ord('A') + i)
            tgt = chr(ord('A') + i + 1)
            M, _M, rows, columns = self.get_matrix(src, tgt)
            data[src + tgt] = Munch({'full': M, 'half': _M, 'rank': Matrix(_M).rank(),
                                     'rows': rows, 'columns': columns})
        return data
