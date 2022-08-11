from graph_parent import GraphParent
from utils import parallel_loop
from datetime import datetime
from multiprocessing import Process, Manager


class GraphFamily:

    def __init__(self, args, threads, name='A'):
        self.repr = None
        self.no = None
        self.o = None
        self.graphs = []
        self.name = name
        self.threads = threads
        if len(args) > 0:
            if type(args[0]) == list:
                for graph in args:
                    self.graphs.append(GraphParent(graph, threads))
            else:
                self.graphs = args
        self.graphs.sort()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]

    def find_invar(self, i):
        _ = self.graphs[i].sG.invar

    def find_z_and_ori(self, i):
        _ = self.graphs[i].orientable
        _ = self.graphs[i].Zall

    def find_unique_invar(self, i, cores, return_dict):
        start = i * int(len(self.graphs) / cores)
        end = min(len(self.graphs), (i + 1) * int(len(self.graphs) / cores))
        if i + 1 == cores:
            end = len(self.graphs)
        invar_list = []
        repr_list = []
        for i in range(start, end):
            if self.graphs[i].sG.invar not in invar_list:
                invar_list.append(self.graphs[i].sG.invar)
                repr_list.append(self.graphs[i])

        return_dict[i] = repr_list

    def set_repr_task(self, i, cores, invar_list, repr_list):
        start = i * int(len(self.graphs) / cores)
        end = min(len(self.graphs), (i + 1) * int(len(self.graphs) / cores))
        if i + 1 == cores:
            end = len(self.graphs)
        for i in range(start, end):
            self.graphs[i].repr = repr_list[invar_list.index(self.graphs[i].sG.invar)]

    def set_repr(self):
        print(f"{datetime.now()}, Computing invariant for {self.name} graphs")
        _ = parallel_loop(self.find_invar, len(self.graphs), self.threads)

        print(f"{datetime.now()}, Finding unique invariant for {self.name} graphs")

        # Sequential execution
        # invar_list = []
        # repr_list = []
        # for g in self.graphs:
        #     if g.sG.invar not in invar_list:
        #         invar_list.append(g.sG.invar)
        #         repr_list.append(g)
        #         g.repr = g
        #         g.is_repr = True
        #     else:
        #         g.repr = repr_list[invar_list.index(g.sG.invar)]

        # Parallel execution, find representatives
        repr_list = []
        invar_list = []
        cores = min(self.threads, int(len(self.graphs) / 2))
        with Manager() as manager:
            return_dict = manager.dict()
            jobs = []
            for p in range(cores):
                jobs.append(Process(target=self.find_unique_invar, args=(p, cores, return_dict,)))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            for sub_list in list(return_dict.values()):
                for g in sub_list:
                    if g.sG.invar not in invar_list:
                        invar_list.append(g.sG.invar)
                        repr_list.append(g)
                        g.is_repr = True
            jobs = []
            cores = min(self.threads, len(self.graphs))
            for p in range(cores):
                jobs.append(Process(target=self.set_repr_task, args=(p, cores, invar_list, repr_list, )))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()

        print(f"{datetime.now()}, Computing orientability and Zs for {self.name} graphs")
        if self.name != 'A':
            _ = parallel_loop(self.find_z_and_ori, len(self.graphs), self.threads)

        print(f"{datetime.now()}, Grouping {self.name} representatives by orientability")
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

        print(f"{datetime.now()}, Giving name for {self.name} representatives")
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

        print(f"{datetime.now()}, Found {len(self.o)} of {self.name} graphs and {len(self.no)} of {self.name}N graphs")

    def find_deeper_graphs(self, i):
        return self.repr[i].er_sets

    def deeper_graphs(self):
        print(f"{datetime.now()}, Searching {chr(ord(self.name) + 1)} graphs...")
        results = parallel_loop(self.find_deeper_graphs, len(self.repr), self.threads)
        data = []
        for _data in results:
            data.extend(_data)
        print(f"{datetime.now()}, Found {len(data)} of {chr(ord(self.name) + 1)} graphs")
        return GraphFamily(data, self.threads, chr(ord(self.name) + 1))

    def export_graphs(self, directory):
        print(f"{datetime.now()}, Exporting {self.name} graphs")
        with open(f"{directory}/{self.name}_graphs.txt", "w") as f:
            for prefix, gs in [[self.name, self.o], [self.name + 'N', self.no]]:
                for cnt, g in enumerate(gs, 1):
                    f.write(f"{prefix + str(cnt):>4s}: {g.sG.edges}\n")

    def isolated(self):
        for g in self.graphs:
            del g.src
