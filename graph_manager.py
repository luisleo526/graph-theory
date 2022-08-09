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
            print(
                f"There are {len(next_graphs)} of {next_type} graphs from {len(getattr(self.graphs, last_type).repr)} "
                f"of {last_type} representatives.")
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