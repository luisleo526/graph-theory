import argparse
from graph import GraphSets
import pandas as pd
import numpy as np
import math
import functools


def to_excel(t):
    columns = []
    for i in range(len(t.A.o)):
        columns.append(f"A{i + 1}")
    for i in range(len(t.A.no)):
        columns.append(f"AN{i + 1}")

    rows = []
    for i in range(len(t.B.o)):
        rows.append(f"B{i + 1}")
    for i in range(len(t.B.no)):
        rows.append(f"BN{i + 1}")

    data = np.empty((len(rows), len(columns)), dtype=object)
    data2 = np.empty((len(rows), len(columns)), dtype=object)
    for i in range(len(rows)):
        for j in range(len(columns)):
            tmp = []
            for k in range(len(t.A[0].G)):
                tmp.append(0)
            data[i, j] = tmp
            data2[i, j] = tmp

    sign = functools.partial(math.copysign, 1)
    for g in t.B.cand:
        if 'N' in g.src_graph.name:
            i = len(t.A.o) + int(g.src_graph.name[2:])
        else:
            i = int(g.src_graph.name[1:])

        k = g.src_graph.G.index(g.reduced_edge)

        if 'N' in g.repr.name:
            j = len(t.B.o) + int(g.repr.name[2:])
        else:
            j = int(g.repr.name[1:])

        if type(g.z_repr) == str:
            data[j - 1, i - 1][k] = [u"\u00B11", sign(g.z_sg), sign(g.z_src)]
            data2[j - 1, i - 1][k] = "#"
        else:
            data[j - 1, i - 1][k] = [sign(g.z_repr), sign(g.z_sg), sign(g.z_src)]
            data2[j - 1, i - 1][k] = sign(g.z_repr) * sign(g.z_sg) * sign(g.z_src)

    writer = pd.ExcelWriter(f"n={args.n}.xlsx", engine='xlsxwriter')

    pd.DataFrame(data=data, index=rows, columns=columns).to_excel(excel_writer=writer, sheet_name="ALL")
    pd.DataFrame(data=data2, index=rows, columns=columns).to_excel(excel_writer=writer, sheet_name="ALL-to-One")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2, type=int)
    parser.add_argument("-t", default=8, type=int)
    args = parser.parse_args()

    t = GraphSets(n=args.n, threads=args.t)
    to_excel(t)
