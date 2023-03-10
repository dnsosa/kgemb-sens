"""General utilities for all code."""

# -*- coding: utf-8 -*-

import math

import networkx as nx


def good_round(x):
    frac = x - math.floor(x)
    if frac < 0.5:
        return math.floor(x)
    return math.ceil(x)


def net2df(G):
    return nx.to_pandas_edgelist(G, edge_key="key")


def retrieve_rel_whitelist(dataset, task):
    """
    Retrieve the whitelist set of relations based on the input KG dataset and the prediction task

    :param dataset: KG dataset
    :param task: prediction task
    """
    dataset_task_mapper = {
        "gnbr":
            {"DrDz": {"T"},
             "DrG": {"A+", "A-", "N"},  # TO DO: Note that can't use 'E' etc. because it's present in two subsets
             "GG": {"W", "V+", "I", "H", "Rg"},
             "DzG": {"U", "Ud", "J", "Y", "G", "X", "L"}
             },
        "hetionet":
            {"DrDz": {"CtD"},
             "DrG": {"CbG"},
             "GG": {"GiG"},
             "DzG": {"DaG"}
             },
    }

    if "gnbr" in dataset:
        dataset_id = "gnbr"
    elif "hetionet" in dataset:
        dataset_id = "hetionet"

    dataset_mapper = dataset_task_mapper.get(dataset_id, {})
    whitelist_rels = dataset_mapper.get(task, None)

    print(f"Given the dataset: {dataset} and the task {task}, these rels were retrieved -- {whitelist_rels}")
    return whitelist_rels
