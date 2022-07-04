import argparse
from graph import GraphSets
import multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2, type=int)
    parser.add_argument("-t", default=1, type=int)
    args = parser.parse_args()

    pool = mp.Pool(processes=args.t)
    g = GraphSets(n=args.n, pool=pool)

    with open(f"{args.n}-results.txt", 'w') as f:
        for i, graph in enumerate(g.graphs):
            f.write(f"A({args.n*2}, {i+1})\n")
            f.write(graph.infos())
            f.write("="*30+"\n\n")
            graph.release_all_memory()
