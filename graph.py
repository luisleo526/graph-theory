import numpy as np
from sympy.matrices import Matrix
from sympy import symbols, Poly


class Graph:

    def __init__(self, n, graph):
        self.edges = graph

        self.adj = np.zeros((2 * n, 2 * n))
        for (i, j) in self.edges:
            self.adj[i - 1, j - 1] = 1
        self.adj += self.adj.transpose()

        x = symbols('x')
        self.poly = Poly(Matrix(np.identity(2 * n) * x + self.adj).det(), x)
        self.coeffs = self.poly.all_coeffs()


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

        self.graphs = []
        for _graph in _graphs:
            graph = []
            for edge in _graph:
                vertices = edge.replace(' :', '').split(' ')
                for i in range(1, len(vertices)):
                    if int(vertices[i]) > int(vertices[0]):
                        graph.append((int(vertices[0]), int(vertices[i])))
            self.graphs.append(Graph(n=n, graph=graph))
