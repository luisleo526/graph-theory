from graph_family import GraphFamily
from utils import readGraph, get_data
from pathlib import Path
import argparse
import pandas as pd
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=4, type=int)
    parser.add_argument("-t", default=8, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    Path(f"./{args.n}_graphs").mkdir(parents=True, exist_ok=True)

    src_graphs = GraphFamily(readGraph(args.n), threads=args.t)
    for g in src_graphs:
        g._orientable = True
    src_graphs.set_repr()
    src_graphs.export_graphs(f"./{args.n}_graphs")

    while True:
        tgt_graphs = src_graphs.deeper_graphs()
        if len(tgt_graphs) > 0:
            rows, columns, data, rank = get_data(src_graphs, tgt_graphs)
            with pd.ExcelWriter(f"./{args.n}_graphs/{src_graphs.name + tgt_graphs.name}.xlsx") as writer:
                pd.DataFrame(data=data.transpose(), index=rows, columns=columns).to_excel(writer, sheet_name='Matrix')
                pd.DataFrame(data={'rank': [rank]}).to_excel(writer, sheet_name='Ranks')
        else:
            break
        tgt_graphs.isolated()
        del src_graphs
        src_graphs = tgt_graphs
