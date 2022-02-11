# -*- coding: utf-8 -*-

"""Visualizations of the networks post-perturbation."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph_nice(G, train_subset, test_subset, sparse_subset=None, new_contradictions=None,
                    removed_contradictions=None):
    np.random.seed(1238)
    pos = nx.spring_layout(G)

    # Fig 1, show edge labels and "flattened" graph
    plt.figure(figsize=(6, 6))

    nx.draw_networkx_nodes(G, pos=pos, node_color="#b0b0b0", node_size=20, alpha=.4)
    nx.draw_networkx_labels(G, pos=pos, font_color="green", font_size=16)

    ###nx.draw_networkx_edges(G, pos=pos, edge_color="#b0b0b0", width=.3, alpha=.5)
    nx.draw_networkx_edges(G, pos=pos, edgelist=train_subset, edge_color="#d6c618", width=.6, alpha=1)  # yellow
    nx.draw_networkx_edges(G, pos=pos, edgelist=test_subset, edge_color="blue", width=.4, alpha=1)
    if sparse_subset is not None:
        nx.draw_networkx_edges(G, pos=pos, edgelist=sparse_subset, edge_color="#cccccc", width=.5, alpha=.5,
                               style=':')  # light gray
    if new_contradictions is not None:
        nx.draw_networkx_edges(G, pos=pos, edgelist=new_contradictions, edge_color="#700000", width=1.5,
                               alpha=.5)  # dark blood red
    if removed_contradictions is not None:
        nx.draw_networkx_edges(G, pos=pos, edgelist=removed_contradictions, edge_color="white", width=2, alpha=1,
                               style=':')

        ### edge_labels = dict([((u,v), r['edge']) for u, v, rd in G1.edges(data=True)])
    e_label_dict = {}
    e_label_dict_nice = {}
    for u, v, _, r in train_subset + test_subset:
        if (u, v) not in e_label_dict.keys():
            e_label_dict[(u, v)] = [r['edge']]
        else:
            e_label_dict[(u, v)].append(r['edge'])

    for u, v in e_label_dict.keys():
        e_label_dict_nice[(u, v)] = '\n'.join(sorted(list(set(e_label_dict[(u, v)]))))
    nx.draw_networkx_edge_labels(G, pos=pos, font_size=7, edge_labels=e_label_dict_nice)

    plt.axis('off')
    plt.show()
