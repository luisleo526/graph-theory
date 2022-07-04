import argparse
from graph import GraphSets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2, type=int)
    parser.add_argument("-t", default=1, type=int)
    args = parser.parse_args()

    g = GraphSets(n=args.n, threads=args.t)

    with open(f"{args.n}-results.txt", 'w') as f:
        for graph in g.graphs:
            f.write(graph.infos())
