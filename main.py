from graph_manager import GraphManager
import argparse
import pandas as pd
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=4, type=int)
    parser.add_argument("-t", default=8, type=int)
    args = parser.parse_args()

    sys.setrecursionlimit(1000000)

    p = GraphManager(args.n, args.t)
    data = p.get_all_data()

    with pd.ExcelWriter(f"n={args.n}.xlsx") as writer:
        ranks = {'rank': []}
        for d in data:
            pd.DataFrame(data=data[d].full.transpose(), index=data[d].rows, columns=data[d].columns).to_excel(writer, sheet_name=d)
            ranks['rank'].append(data[d].rank)
        pd.DataFrame(data=ranks, index=list(data.keys())).to_excel(writer, sheet_name='ranks')

    with open(f"n={args.n}-graphs.txt", "w") as f:
        for i in range(p.maxi + 1):
            graphs = getattr(p, chr(ord('A') + i))
            for prefix, gs in [[graphs.name, graphs.o], [graphs.name + 'N', graphs.no]]:
                for cnt, g in enumerate(gs, 1):
                    f.write(f"{prefix + str(cnt)}: {g.sG.edges}\n")
