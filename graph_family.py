from graph_parent import GraphParent
from utils import parallel_loop


class GraphFamily:

    def __init__(self, args, threads, name='A'):
        self.repr = None
        self.no = None
        self.o = None
        self.graphs = []
        self.name = name
        self.threads = threads
        if type(args[0]) == list:
            for graph in args:
                self.graphs.append(GraphParent(graph, threads))
        else:
            self.graphs = args

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]

    def find_invar(self, i):
        _ = self.graphs[i].sG.invar

    def set_repr(self):
        # Precompute invariant
        _ = parallel_loop(self.find_invar, len(self.graphs), self.threads)

        # Find representative
        invar_list = []
        repr_list = []
        for g in sorted(self.graphs):
            if g.sG.invar not in invar_list:
                invar_list.append(g.sG.invar)
                repr_list.append(g)
                g.repr = g
                g.is_repr = True
            else:
                for gr in repr_list:
                    if gr.sG.invar == g.sG.invar:
                        g.repr = gr
                        break
        # Find orientable
        self.o = []
        self.no = []
        cnt1 = 0
        cnt2 = 0
        for g in repr_list:
            if g.orientable:
                self.o.append(g)
                cnt1 += 1
                g.name = self.name + str(cnt1)
            else:
                self.no.append(g)
                cnt2 += 1
                g.name = self.name + "N" + str(cnt2)

        # Set index for representative
        cnt1 = 0
        cnt2 = 0
        for g in repr_list:
            if g.orientable:
                g.id = cnt1
                cnt1 += 1
            else:
                g.id = cnt2 + len(self.o)
                cnt2 += 1

        self.repr = repr_list

    def find_deeper_graphs(self, i):
        return self.repr[i].er_sets

    def deeper_graphs(self):
        results = parallel_loop(self.find_deeper_graphs, len(self.repr), self.threads)
        data = []
        for _data in results:
            data += _data
        return data
