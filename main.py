import argparse
from graph import GraphSets
import pandas as pd
import numpy as np
import math
import functools
import time
from sympy.matrices import Matrix
import multiprocessing as mp


def get_dataframe(t, src, tgt):
    columns = []
    for i in range(len(getattr(t, src).o)):
        columns.append(f"{src}{i + 1}")
    for i in range(len(getattr(t, src).no)):
        columns.append(f"{src}N{i + 1}")

    rows = []
    for i in range(len(getattr(t, tgt).o)):
        rows.append(f"{tgt}{i + 1}")
    for i in range(len(getattr(t, tgt).no)):
        rows.append(f"{tgt}N{i + 1}")
    rows.append("X")

    data = np.empty((len(rows), len(columns)), dtype=object)
    for i in range(len(rows)):
        for j in range(len(columns)):
            data[i, j] = ""

    edges = np.empty(len(columns), dtype=object)
    for i in range(len(columns)):
        edges[i] = []
        for j in range(len(getattr(t, src)[0].G)):
            edges[i].append(0)

    sign = functools.partial(math.copysign, 1)
    for g in getattr(t, tgt):
        if 'N' in g.src_graph.name:
            i = len(getattr(t, src).o) + int(g.src_graph.name[2:])
        else:
            i = int(g.src_graph.name[1:])

        k = g.src_graph.sort.index(g.reduced_edge) + 1

        if 'N' in g.repr.name:
            j = len(getattr(t, tgt).o) + int(g.repr.name[2:])
        else:
            j = int(g.repr.name[1:])

        edges[i - 1][k - 1] = 1

        if type(g.z_repr) == str:
            data[j - 1, i - 1] += f"{'{'}#{k},[{int(sign(g.z_src))}, {int(sign(g.z_sg))}, \u00B11]{'}'}, "
        else:
            data[j - 1, i - 1] += f"{'{'}#{k},[{int(sign(g.z_src))}, {int(sign(g.z_sg))}, {int(sign(g.z_repr))}]{'}'}, "

    for i in range(len(columns)):
        for j in range(len(edges[i])):
            if edges[i][j] == 0:
                data[-1, i] += f"#{j + 1},"

    return pd.DataFrame(data=data, index=rows, columns=columns)


def get_matrix(t, src, tgt):
    columns = []
    for i in range(len(getattr(t, src).o)):
        columns.append(f"{src}{i + 1}")
    for i in range(len(getattr(t, src).no)):
        columns.append(f"{src}N{i + 1}")

    rows = []
    for i in range(len(getattr(t, tgt).o)):
        rows.append(f"{tgt}{i + 1}")
    for i in range(len(getattr(t, tgt).no)):
        rows.append(f"{tgt}N{i + 1}")

    data = np.empty((len(rows), len(columns)), dtype=int)
    for i in range(len(rows)):
        for j in range(len(columns)):
            data[i, j] = 0

    sign = functools.partial(math.copysign, 1)
    for g in getattr(t, tgt):
        if 'N' in g.src_graph.name:
            i = len(getattr(t, src).o) + int(g.src_graph.name[2:])
        else:
            i = int(g.src_graph.name[1:])

        if 'N' in g.repr.name:
            j = len(getattr(t, tgt).o) + int(g.repr.name[2:])
        else:
            j = int(g.repr.name[1:])

        data[j - 1, i - 1] += g.z_mult

    return pd.DataFrame(data=data, index=rows, columns=columns), data[:len(getattr(t, tgt).o), :len(getattr(t, src).o)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=4, type=int)
    parser.add_argument("-t", default=8, type=int)
    args = parser.parse_args()

    cpu_time = -time.time()

    t = GraphSets(n=args.n, threads=args.t)

    with pd.ExcelWriter(f"n={args.n}-details.xlsx") as writer:
        for i in range(26):
            get_dataframe(t, chr(65 + i), chr(65 + i + 1)).to_excel(writer,
                                                                    sheet_name=f"{chr(65 + i)}-{chr(65 + i + 1)}")
            if len(getattr(t, chr(65 + i + 2))) == 0:
                break

    matrices = {}
    with pd.ExcelWriter(f"n={args.n}-matrix.xlsx") as writer:
        for i in range(26):
            df,  matrix = get_matrix(t, chr(65 + i), chr(65 + i + 1))
            matrices[f"{chr(65 + i)}{chr(65 + i+1)}"] = matrix
            df.to_excel(writer, sheet_name=f"{chr(65 + i)}-{chr(65 + i + 1)}")
            if len(getattr(t, chr(65 + i + 2))) == 0:
                break

    with open(f"n={args.n}-graphs.txt", "w") as f:
        for i in range(26):
            if len(getattr(t, chr(65+i))) > 0:
                for g in getattr(t, chr(65+i)).o:
                    f.write(f"{g.name}: {g.sort}\n")
                for g in getattr(t, chr(65+i)).no:
                    f.write(f"{g.name}: {g.sort}\n")
            else:
                break

    with open(f"n={args.n}-matrices_ranks.txt", "w") as f:
        for t in matrices:
            if matrices[t].size > 0:
                f.write(f"{t}: {Matrix(matrices[t]).rank()}\n")
            else:
                f.write(f"{t}: 0\n")

    print("CPU time:", time.time()+cpu_time)
