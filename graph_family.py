from graph_parent import GraphParent
from utils import parallel_loop
from datetime import datetime
from multiprocessing import Process, Manager


class GraphFamily:

    def __init__(self, args, threads, name='A'):
        self.invar = None
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
        return i, self.graphs[i].sG.invar

    def find_z_and_ori(self, i):
        return i, self.graphs[i].orientable, self.graphs[i].Zh, self.graphs[i].Zr, self.graphs[i].Zs

    def find_unique_invar(self, p, cores, return_dict):
        start = p * int(len(self.graphs) / cores)
        end = min(len(self.graphs), (p + 1) * int(len(self.graphs) / cores))
        if p + 1 == cores:
            end = len(self.graphs)
        invar_list = []
        repr_list = []
        for i in range(start, end):
            if self.graphs[i].sG.invar not in invar_list:
                invar_list.append(self.graphs[i].sG.invar)
                repr_list.append(i)

        return_dict[p] = repr_list

    def set_repr_task(self, i):
        return i, self.invar.index(self.graphs[i].sG.invar)

    def split_repr(self, p, cores, return_dict):
        start = p * int(len(self.repr) / cores)
        end = min(len(self.repr), (p + 1) * int(len(self.repr) / cores))
        if p + 1 == cores:
            end = len(self.repr)
        o = []
        no = []
        for i in range(start, end):
            if self.repr[i].orientable:
                o.append(i)
            else:
                no.append(i)
        return_dict[p] = (o, no)

    def set_repr(self):
        print(f"{datetime.now()}, Computing invariant for {self.name} graphs")
        results = parallel_loop(self.find_invar, len(self.graphs), self.threads)
        for i, invar in results:
            self.graphs[i]._invar = invar

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
        cores = min(self.threads, max(1, int(len(self.graphs) / 2)))
        with Manager() as manager:
            return_dict = manager.dict()
            jobs = []
            print(f"{datetime.now()}, Finding unique invariant for {self.name} graphs")
            for p in range(cores):
                jobs.append(Process(target=self.find_unique_invar, args=(p, cores, return_dict,)))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            for sub_list in list(return_dict.values()):
                for i in sub_list:
                    if self.graphs[i].sG.invar not in invar_list:
                        invar_list.append(self.graphs[i].sG.invar)
                        repr_list.append(self.graphs[i])

        self.repr = repr_list
        self.invar = invar_list

        print(f"{datetime.now()}, Setting representatives for {self.name} graphs")
        results = parallel_loop(self.set_repr_task, len(self.graphs), self.threads)
        for i, j in results:
            self.graphs[i].repr = self.repr[j]

        print(f"{datetime.now()}, Computing orientability and Zs for {self.name} graphs")
        if self.name != 'A':
            results = parallel_loop(self.find_z_and_ori, len(self.graphs), self.threads)
            for i, ori, zh, zr, zs in results:
                self.graphs[i]._orientable = ori
                self.graphs[i]._Zh = zh
                self.graphs[i]._Zr = zr
                self.graphs[i]._Zs = zs

        print(f"{datetime.now()}, Grouping {self.name} representatives by orientability")
        self.o = []
        self.no = []
        cores = min(self.threads, max(1, int(len(self.repr) / 2)))
        with Manager() as manager:
            return_dict = manager.dict()
            jobs = []
            for p in range(cores):
                jobs.append(Process(target=self.split_repr, args=(p, cores, return_dict,)))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            for o, no in list(return_dict.values()):
                self.o.extend([self.repr[i] for i in o])
                self.no.extend([self.repr[i] for i in no])

        print(f"{datetime.now()}, Giving name, id for {self.name} representatives")
        for s, prefix, _list in [[0, self.name, self.o], [len(self.o), self.name + 'N', self.no]]:
            for cnt, g in enumerate(_list, 1):
                g.name = prefix + str(cnt + s)
                g.id = cnt - 1 + s

        print(f"{datetime.now()}, Found {len(self.o)}/{len(self.no)} for {self.name}/{self.name}N graphs")

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
