from graph_parent import GraphParent


class GraphFamily:

    def __init__(self, args, threads, name='A'):
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

    def set_repr(self):
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

    def deeper_graphs(self):
        data = []
        for g in self.graphs:
            if g.is_repr:
                data += g.er_sets
        return data
