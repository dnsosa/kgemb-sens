# -*- coding: utf-8 -*-

"""Initial pre-processing of KGs before experimental perturbation."""

from ..transform.graph_utilities import get_lcc


def preprocess_network_pipeline(G, dataset, do_get_lcc):
    """
    Pre-process the input KG before any experimental perturbation.

    :param G: input KG
    """
    if dataset == "gnbr":
        G = clean_gnbr_dataset(G)

    if do_get_lcc:
        # TODO: Check if get_lcc is doing what we want/expect
        G = get_lcc(G)

    return G


def clean_gnbr_dataset(G):
    """
    Clean GNBR by removing very hubby nodes that are likely just noise.

    :param G: input (GNBR) KG
    """
    print("\nCleaning the full GNBR network...")
    print(f"Num. nodes before cleanup: {G.number_of_nodes()}")

    G_degree_dict = dict(G.degree())

    # Clean the drug list
    banned_drugs = [node for node in G.nodes() if
                    ("Compound::MESH" in node) or (("Compound" in node) and (G_degree_dict[node] > 500))]
    G.remove_nodes_from(banned_drugs)
    print(f"Removing {len(banned_drugs)} sketchy drugs.")

    # Clean the disease list
    banned_diseases = [node for node in G.nodes() if ("Disease" in node) and (G_degree_dict[node] > 1000)]
    G.remove_nodes_from(banned_diseases)
    print(f"Removing {len(banned_diseases)} sketchy diseases.")

    print(f"Num. nodes after cleanup: {G.number_of_nodes()} .")

    return G
