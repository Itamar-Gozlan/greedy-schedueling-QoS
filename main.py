import argparse
import colorsys
import copy

import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt


def get_colors(num_colors):
    """
    :param num_colors: How many distinct colors to generate
    :return: list with num_colors distinct colors.
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def generate_stage_vector_DAG(n, m, type):
    """
    :param n: Number of stages
    :param m: Number of phases that will to execute in a pipeline
    :param type: Stage phase processing time distribution
    :return: The DAG edges list, positions for the DAG plot and dictionary stage to phase vector. The ith place in the
    vector indicate the required processing time for the ith phase
    """
    G = nx.generators.directed.gnc_graph(n)
    if type == 'random':
        stage_vec = {i: [2 + np.random.rand() * 5 for k in range(m)] for i in range(n)}
    else:
        stage_vec = {i: [1 for k in range(m)] for i in range(n)}
    S = set(G.nodes())
    removed_edges = []
    curr_x = -1
    pos = {node: None for node in G.nodes}
    while len(S) > 0:
        curr_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes))
        y_ax = np.linspace(-1, 1, len(curr_nodes)) if len(curr_nodes) > 1 else [0]
        for node, y in zip(curr_nodes, y_ax):
            pos[node] = (curr_x, y)
            edges_to_remove = []
            for nbr in G.neighbors(node):
                edge = (node, nbr)
                edges_to_remove.append(edge)
            for e in edges_to_remove:
                G.remove_edge(*e)
            G.remove_node(node)
            S.remove(node)

            removed_edges += edges_to_remove
        curr_x += 0.2
    return removed_edges, pos, stage_vec


def greedy_scheduling(G, stage_vec):
    """
    Schedule n stages s.t each stage has m phases that should be executed in a pipeline.
    :param G: DiGraph (DAG) with stages precedence
    :param stage_vec: vector for each stage with phase processing time
    :return: dictionary with scheduling for each machine
    """
    S = set(G.nodes())
    t_hat = [0, 0, 0, 0, 0]
    scheduling = {}
    max_logger = []
    while len(S) > 0:
        ready2go = list(filter(lambda node: G.in_degree(node) == 0, G.nodes))
        t_hat = [max(t_hat)]*len(t_hat)
        max_logger.append(max(t_hat))
        for node in ready2go:
            state_sched = [0, 0, 0, 0, 0]
            state_sched[0] = t_hat[0]
            t_hat[0] = t_hat[0] + stage_vec[node][0]
            for j in range(1, len(t_hat)):
                state_sched[j] = max(t_hat[j - 1],
                                     t_hat[j])
                t_hat[j] = state_sched[j] + stage_vec[node][j]
            scheduling[node] = copy.deepcopy(state_sched)

        S = S - set(ready2go)
        G.remove_nodes_from(ready2go)

    return scheduling, max_logger


def plot_scheduling(scheduling, stage_vec, args, max_logger):
    """
    Plot function for the scheduling algorithm
    :param scheduling: Dictionary with scheduling (start times) for each machine
    :param stage_vec: Dictionary of stage->(p1,p2,..,pm) phases processing time
    :param m: number of machines in the system (should be equal to the phases of the stages)
    :return: None. Plot
    """
    m = args.m
    fig, ax = plt.subplots()
    ax.set_xlabel('Time Unit')
    ax.set_ylabel('Machine')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_yticks([5 + i for i in range(0, m * 10, 10)], labels=list(range(m)))
    node_to_color = get_colors(len(stage_vec))
    machine_sched = {i: [] for i in range(m)}
    machine_color = {i: [] for i in range(m)}
    stage_annotations = {i: [] for i in scheduling}
    for node in stage_vec.keys():
        for machine, phase_start in enumerate(scheduling[node]):
            machine_sched[machine].append((phase_start, stage_vec[node][machine]))
            stage_annotations[node].append(((phase_start + stage_vec[node][machine]/2), machine * 10 + 5))
            machine_color[machine].append(node_to_color[node])

    for machine, sched in machine_sched.items():
        sorted_sched = sorted(sched, key=lambda x: x[0])
        ax.broken_barh(sorted_sched, (machine * 10, 9), facecolors=machine_color[machine], edgecolor='black')

    for sched in machine_sched.values():
        sorted_sched = sorted(sched, key=lambda x: x[0])
        s, e = sorted_sched[0]
        for curr_s, curr_e in sorted_sched[1:]:
            if curr_s < e:
                raise ValueError("Scheduling overlap detected!!!")
                e = curr_e

    for stage in stage_annotations:
        for pnt in stage_annotations[stage]:
            ax.annotate(stage, pnt)

    plt.vlines(x=max_logger, ymin=0, ymax=args.m*10, colors='black', ls='--', lw=2)

    ax.legend()
    fig.savefig('figures/sched_{0}.jpg'.format(args.type))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Number of states for the experiment', type=int, default=10)
    parser.add_argument('-m', help='Number of machines for the experiment', type=int, default=5)
    parser.add_argument('-type', help='Phases processing time. \"random\" or \"constant\"', type=str, default="random")
    parser.add_argument('-plot_dag', help='True/False plot of the random DAG graph', type=bool, default=True)
    parser.add_argument('-plot_scheduling', help='True/False plot of scheduling graph', type=bool, default=True)
    args = parser.parse_args()

    edges, pos, stage_vec = generate_stage_vector_DAG(args.n, args.m, args.type)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    scheduling, max_logger = greedy_scheduling(G, stage_vec)
    if args.plot_dag:
        G.add_edges_from(edges)  # all nodes were removed and should be constructed again
        nx.draw(G, pos=pos, with_labels=True)
        plt.savefig('figures/dag_{0}.jpg'.format(args.type))

    if args.plot_scheduling:
        plot_scheduling(scheduling, stage_vec, args, max_logger)

    if args.plot_dag or args.plot_scheduling:
        plt.show()


if __name__ == "__main__":
    main()
