from graph_parent import GraphParent
from utils import parallel_loop
from datetime import datetime
from multiprocessing import Process, Manager
import random
import pickle
from collections import defaultdict


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

        data = defaultdict(list)
        for i in range(start, end):
            data[self.graphs[i].sG.invar].append(i)

        return_dict[p] = data

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

        print(f"{datetime.now()}, Finding unique invariant for {self.name} graphs")

        self.invar = defaultdict(list)
        cores = min(self.threads, max(1, int(len(self.graphs) / 128)))
        with Manager() as manager:
            return_dict = manager.dict()
            jobs = []
            for p in range(cores):
                jobs.append(Process(target=self.find_unique_invar, args=(p, cores, return_dict,)))
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            for _dict in list(return_dict.values()):
                for key, value in _dict.items():
                    self.invar[key].extend(value)

        self.repr = []
        for invar in self.invar:
            self.invar[invar].sort()
            self.repr.append(self.graphs[self.invar[invar][0]])
            for index in self.invar[invar]:
                self.graphs[index].repr = self.graphs[self.invar[invar][0]]

        print(f"{datetime.now()}, Computing orientability and Zh, Zs, Zr for {self.name} graphs")
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
        cores = min(self.threads, max(1, int(len(self.repr) / 1024)))
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
                    f.write(f"{prefix + str(cnt):>8s}: {g.sG.edges}\n")

    def export_to_binary(self, directory):
        print(f"{datetime.now()}, Exporting {self.name} object to binary")
        with open(f"{directory}/{self.name}", "wb") as f:
            pickle.dump(self, f)

    def isolated(self):
        for g in self.graphs:
            g.src = g.src.name
