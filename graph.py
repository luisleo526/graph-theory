import numpy as np
from sympy.matrices import Matrix
from sympy import symbols, LC
from sympy.utilities.iterables import multiset_permutations

class Graph:

    def __init__(self, n, graph):
        self.edges = graph

        # Create adjacency matrix
        self.adj = np.zeros((2 * n, 2 * n))
        for (i, j) in self.edges:
            self.adj[i - 1, j - 1] = 1
        self.adj += self.adj.transpose()

        # Calculate invariant
        self.invar = []
        self.invar.append(Matrix(self.adj).charpoly(symbols('x')).as_expr())
        for i in range(2 * n):
            _adj = np.delete(np.delete(self.adj, i, 0), i, 1)
            self.invar.append(Matrix(_adj).charpoly(symbols('x')).as_expr())

        # Compare polynomials
        diffs = []
        for i in range(len(self.invar)):
            ans = []
            for j in range(len(self.invar)):
                diff = self.invar[i] - self.invar[j]
                if len(diff.free_symbols) == 0:
                    ans.append(0)
                else:
                    ans.append(LC(diff))
            diffs.append(sum([1 for x in ans if x > 0]))

        # Standardization
        self.permute_index = []
        for e in sorted(set(diffs)):
            vertices = []
            for i in range(len(diffs)):
                if diffs[i] == e:
                    vertices.append(i)
            self.permute_index.append(vertices)

        # Find all permutations
        self.permutation_sets = []
        for index_set in self.permute_index[:-1]:
            self.permutation_sets.append(list(multiset_permutations(index_set)))


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

    def get_graph_info(self, i):
        print("Graph:", self.graphs[i].edges)
        print("Invariant:")
        for j, poly in enumerate(self.graphs[i].invar):
            print(f"P(G,{j})", poly)
        print("Ordered Dictionary:", self.graphs[i].permute_index[:-1])
        print(f"Permutation sets (Total {np.prod([len(sets) for sets in self.graphs[i].permutation_sets])}): ")
        for j, permutation in enumerate(self.graphs[i].permutation_sets):
            print(f"Set {j}:", permutation)

    def get_number_of_graphs(self):
        return len(self.graphs)
