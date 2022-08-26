import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
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
    rank_tri = []
    old_half = None

    while True:
        if args.find_binary and os.path.exists(f"./{args.n}_graphs/binary/{chr(ord(src_graphs.name) + 1)}"):
            print(f"{datetime.now()}, Reading from ./{args.n}_graphs/binary/{chr(ord(src_graphs.name) + 1)}")
            tgt_graphs_data = load_from_binary(f"./{args.n}_graphs/binary/{chr(ord(src_graphs.name) + 1)}")
            tgt_graphs = GraphFamily([], args.t, chr(ord(src_graphs.name) + 1))
            tgt_graphs.inherit(src_graphs, tgt_graphs_data)
        else:
            tgt_graphs = src_graphs.deeper_graphs()
            if len(tgt_graphs) == 0:
                print('Ranks:', all_ranks)
                print('Tri-Ranks', rank_tri)
                exit()
            tgt_graphs.set_repr()
        tgt_graphs.export_graphs(f"./{args.n}_graphs")

        rows, columns, details, infos, ranks = get_data(src_graphs, tgt_graphs, args.t, args.n)
        # (rows, tri_rows), (columns, tri_cols), (details, tri_details), (data, half_data, tri_data), rank
        all_ranks.append(ranks[0])
        rank_tri.append(ranks[1])

        print(f"{datetime.now()}, Exporting data for {src_graphs.name + tgt_graphs.name} matrix")

        # --------------------------------------------------------------------------------------
        # All details
        data = defaultdict(list)
        for i, j in np.transpose(np.nonzero(details[0])):
            data[i].append(j)
        with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}.txt", "w") as f:
            f.write(f"{src_graphs.name}: {len(src_graphs.o)}/{len(src_graphs.no)}\n")
            f.write(f"{tgt_graphs.name}: {len(tgt_graphs.o)}/{len(tgt_graphs.no)}\n")
            f.write(f"Rank: {ranks[0]}\n")
            f.write(f"# of ZeroColumns (full): {len(np.where(~infos[0].any(axis=1))[0])}\n")
            f.write(f"Column ID (half): {[x + 1 for x in list(np.where(~infos[0].any(axis=1))[0])]}\n")
            f.write("-" * 40 + "\n")
            for i in sorted(list(data.keys())):
                data[i].sort()
                for j in data[i]:
                    if j != details[0].shape[1] - 1:
                        f.write(f"({columns[0][i]},{rows[0][j]}): {infos[0][i, j]} >> {details[0][i, j]}\n")
                    else:
                        f.write(f"({columns[0][i]}, X): {details[0][i, j]}\n")
                f.write("-" + "\n")

        # --------------------------------------------------------------------------------------
        # All details of transpose
        data = defaultdict(list)
        for i, j in np.transpose(np.nonzero(details[0])):
            if j != details[0].shape[1] - 1:
                data[j].append(i)
        unbind_data = np.sum(details[0].T[:-1] != "", axis=1)
        with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}_invert.txt", "w") as f:
            f.write(f"{src_graphs.name}: {len(src_graphs.o)}/{len(src_graphs.no)}\n")
            f.write(f"{tgt_graphs.name}: {len(tgt_graphs.o)}/{len(tgt_graphs.no)}\n")
            f.write(f"Rank: {ranks[0]}\n")
            f.write(f"# of ZeroColumns (full): {len(np.where(~infos[0].any(axis=1))[0])}\n")
            f.write(f"Column ID (half): {[x + 1 for x in list(np.where(~infos[1].any(axis=1))[0])]}\n")
            f.write("-" * 40 + "\n")
            for j in sorted(list(data.keys())):
                data[j].sort()
                if j >= len(tgt_graphs.o):
                    tgt_graphs.no[j - len(tgt_graphs.o)].unbind_from_data = unbind_data[j]
                    msg = f"{rows[0][j]} unbind numbers (data, computed): (" \
                          f"{tgt_graphs.no[j - len(tgt_graphs.o)].unbind_from_data}," \
                          f"{tgt_graphs.no[j - len(tgt_graphs.o)].unbind})"
                else:
                    tgt_graphs.o[j].unbind_from_data = unbind_data[j]
                    msg = f"{rows[0][j]} unbind numbers (data, computed): ({tgt_graphs.o[j].unbind_from_data}," \
                          f"{tgt_graphs.o[j].unbind})"
                for i in data[j]:
                    f.write(f"({rows[0][j]}, {columns[0][i]}): {infos[0][i, j]} >> {details[0][i, j]}\n")
                f.write(f"{msg}\n")
                f.write("-" + "\n")

        # --------------------------------------------------------------------------------------
        # Prepare indices
        src_tri = [i for i in range(len(src_graphs.o)) if src_graphs.o[i].sG.has_triangle]
        src_notri = [i for i in range(len(src_graphs.o)) if not src_graphs.o[i].sG.has_triangle]
        tgt_tri = [i for i in range(len(tgt_graphs.o)) if tgt_graphs.o[i].sG.has_triangle]
        tgt_notri = [i for i in range(len(tgt_graphs.o)) if not tgt_graphs.o[i].sG.has_triangle]

        # Set data for triangle sorting

        UL = infos[2][:len(src_tri), :len(tgt_tri)]
        UL_details = details[1][:len(src_tri), :len(tgt_tri)]
        UL_indices = defaultdict(list)
        for i, j in np.transpose(np.nonzero(UL)):
            UL_indices[i].append(j)
        with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}_UL.txt", "w") as f:
            f.write(f"Rank (UL, DR): {ranks[1]}\n")
            f.write(f"{src_graphs.name}: {len(src_tri)} / {len(src_notri)}\n")
            f.write(f"{tgt_graphs.name}: {len(tgt_tri)} / {len(tgt_notri)}\n")
            for i in sorted(list(UL_indices.keys())):
                UL_indices[i].sort()
                for j in UL_indices[i]:
                    f.write(f"({columns[1][i]},{rows[1][j]}): {UL[i, j]} >> {UL_details[i, j]}\n")
                f.write("-" + "\n")

        # --------------------------------------------------------------------------------------

        DR = infos[2][-len(src_notri):, -len(tgt_notri):]
        DR_details = details[1][-len(src_notri):, -len(tgt_notri):]
        DR_indices = defaultdict(list)
        for i, j in np.transpose(np.nonzero(DR)):
            DR_indices[i].append(j)
        with open(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}_DR.txt", "w") as f:
            f.write(f"Rank (UL, DR): {ranks[1]}\n")
            f.write(f"{src_graphs.name}: {len(src_tri)} / {len(src_notri)}\n")
            f.write(f"{tgt_graphs.name}: {len(tgt_tri)} / {len(tgt_notri)}\n")
            if len(src_notri) > 0 and len(tgt_notri) > 0:
                for i in sorted(list(DR_indices.keys())):
                    DR_indices[i].sort()
                    for j in DR_indices[i]:
                        f.write(f"({columns[1][i]},{rows[1][j]}): {DR[i, j]} >> {DR_details[i, j]}\n")
                    f.write("-" + "\n")

        print(f"{datetime.now()}, Checking unbind number for {tgt_graphs.name} graphs...")
        checked = True
        for g in tgt_graphs.repr:
            if g.unbind != g.unbind_from_data:
                checked = False
                break
        if checked:
            print("# Pass")
        else:
            print("# Failed.")

        if old_half is not None and infos[1].size > 0:
            print(f"{datetime.now()}, Checking half matrix multiplication for "
                  f"{old_half.name} and {src_graphs.name + tgt_graphs.name}...")
            if np.all(mat_mult(old_half.data, infos[1]) == 0):
                print("# Pass")
            else:
                print("# Failed.")

        old_half = Munch()
        old_half.name = src_graphs.name + tgt_graphs.name
        old_half.data = infos[1]
        tgt_graphs.export_to_binary(f"./{args.n}_graphs/binary")
        del src_graphs
        src_graphs = tgt_graphs
