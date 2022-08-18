import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import os
import numpy as np
import pandas as pd
from munch import Munch
from numba import set_num_threads

from graph_family import GraphFamily
from utils import readGraph, get_data, load_from_binary, mat_mult


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=4, type=int)
    parser.add_argument("-t", default=8, type=int)
    parser.add_argument("-from_graph", default="", type=str)
    parser.add_argument("-file", default="", type=str)
    parser.add_argument("-skip_rank", action='store_true')
    parser.add_argument("-to_excel", action='store_true')
    parser.add_argument("-find_binary", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.file != "":
        args.n = args.file

    set_num_threads(args.t)

    Path(f"./{args.n}_graphs/binary").mkdir(parents=True, exist_ok=True)
    Path(f"./cache").mkdir(parents=True, exist_ok=True)

    if args.from_graph != "":
        print(f"{datetime.now()}, Reading from ./{args.n}_graphs/binary/{args.from_graph}")
        src_graphs = load_from_binary(f"./{args.n}_graphs/binary/{args.from_graph}")
        print(f"{datetime.now()}, Finished reading data")
    elif args.file != "":
        print(f"{datetime.now()}, Reading from ./{args.file}")
        inputs = load_from_binary(args.file)
        src_graphs = GraphFamily(inputs, threads=args.t)
        print(f"{datetime.now()}, Finished reading data")
        for g in src_graphs:
            g._orientable = True
        src_graphs.set_repr()
    else:
        print(f"{datetime.now()}, Reading from GenReg output")
        src_graphs = GraphFamily(readGraph(args.n, args.t), threads=args.t)
        print(f"{datetime.now()}, Finished reading data")
        for g in src_graphs:
            g._orientable = True
        src_graphs.set_repr()
        src_graphs.export_graphs(f"./{args.n}_graphs")
        src_graphs.export_to_binary(f"./{args.n}_graphs/binary")

    all_ranks = []
    old_half = None
    while True:

        if args.find_binary and os.path.exists(f"./{args.n}_graphs/binary/{chr(ord(src_graphs.name) + 1)}"):
            tgt_graphs = load_from_binary(f"./{args.n}_graphs/binary/{chr(ord(src_graphs.name) + 1)}")
            tgt_graphs.link(src_graphs)
        else:
            tgt_graphs = src_graphs.deeper_graphs()
            if len(tgt_graphs) == 0:
                exit()
            tgt_graphs.set_repr()
            tgt_graphs.export_graphs(f"./{args.n}_graphs")

        rows, columns, details, full, half, rank = get_data(src_graphs, tgt_graphs, args.t, args.n, args.skip_rank)
        if not args.skip_rank:
            all_ranks.append(rank)

        print(f"{datetime.now()}, Exporting data for {src_graphs.name + tgt_graphs.name} matrix")

        if args.to_excel:
            with pd.ExcelWriter(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}.xlsx") as writer:
                pd.DataFrame(data=full.transpose(), index=rows, columns=columns).to_excel(writer, sheet_name='Matrix')
                pd.DataFrame(data=details.transpose(),
                             index=rows + ["X"],
                             columns=columns).to_excel(writer, sheet_name='Details')
                if not args.skip_rank:
                    pd.DataFrame(data={'rank': [rank]}).to_excel(writer, sheet_name='Ranks')
                pd.DataFrame(data={'# of ZeroColumns': [len(np.where(~full.any(axis=1))[0])]}).to_excel(
                    writer, sheet_name='ZerosColumns')
                if half.size > 0:
                    pd.DataFrame(data={'Column ID': [x + 1 for x in list(
                        np.where(~half.any(axis=1))[0])]}).to_excel(writer, sheet_name='ZC of Orientable')
        else:
            data = defaultdict(list)
            for i, j in np.transpose(np.nonzero(full)):
                data[i].append(j)
            with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}.txt", "w") as f:
                f.write(f"{src_graphs.name}: {len(src_graphs.o)}/{len(src_graphs.no)}\n")
                f.write(f"{tgt_graphs.name}: {len(tgt_graphs.o)}/{len(tgt_graphs.no)}\n")
                f.write(f"Rank: {rank}\n")
                f.write(f"# of ZeroColumns (full): {len(np.where(~full.any(axis=1))[0])}\n")
                f.write(f"Column ID (half): {[x + 1 for x in list(np.where(~half.any(axis=1))[0])]}\n")
                f.write("-" * 40 + "\n")
                for i in sorted(list(data.keys())):
                    data[i].sort()
                    for j in data[i]:
                        f.write(f"({columns[i]},{rows[j]}): {full[i, j]} >> {details[i, j]}\n")
                    f.write("-")

                data = defaultdict(list)
                for i, j in np.transpose(np.nonzero(full)):
                    data[j].append(i)
                with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}_invert.txt", "w") as f:
                    f.write(f"{src_graphs.name}: {len(src_graphs.o)}/{len(src_graphs.no)}\n")
                    f.write(f"{tgt_graphs.name}: {len(tgt_graphs.o)}/{len(tgt_graphs.no)}\n")
                    f.write(f"Rank: {rank}\n")
                    f.write(f"# of ZeroColumns (full): {len(np.where(~full.any(axis=1))[0])}\n")
                    f.write(f"Column ID (half): {[x + 1 for x in list(np.where(~half.any(axis=1))[0])]}\n")
                    f.write("-" * 40 + "\n")
                    for j in sorted(list(data.keys())):
                        data[j].sort()
                        for i in data[j]:
                            f.write(f"({rows[j]}, {columns[i]}): {full[i, j]} >> {details[i, j]}\n")
                        f.write("-")

        if old_half is not None and half.size > 0:
            print(f"{datetime.now()}, Checking half matrix multiplication for "
                  f"{old_half.name} and {src_graphs.name + tgt_graphs.name}...", end='')
            assert np.all(mat_mult(old_half.data, half) == 0)
            print("Pass")

        old_half = Munch()
        old_half.name = src_graphs.name + tgt_graphs.name
        old_half.data = half
        tgt_graphs.export_to_binary(f"./{args.n}_graphs/binary")
        del src_graphs
        src_graphs = tgt_graphs

    if not args.skip_rank:
        print('Ranks:', all_ranks)
