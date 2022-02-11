# -*- coding: utf-8 -*-

"""For loading various KG datasets."""

import pandas as pd
import networkx as nx

# local data dir
##data_dir="/Users/dnsosa/.data/pykeen/datasets"


def load_data_three_parts(dataset, data_dir="/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"):
    base_dir = f"{data_dir}/{dataset}"

    # Old train/val/test split
    train_path = f"{base_dir}/train.txt"
    val_path = f"{base_dir}/valid.txt"
    test_path = f"{base_dir}/test.txt"

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    train_df.columns = ["source", "edge", "target"]
    train_df = pd.DataFrame({"source": train_df.source, "target": train_df.target, "edge": train_df.edge})

    G_train = nx.from_pandas_edgelist(train_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    val_df = pd.read_csv(val_path, sep="\t", header=None)
    val_df.columns = ["source", "edge", "target"]
    val_df = pd.DataFrame({"source": val_df.source, "target": val_df.target, "edge": val_df.edge})

    G_val = nx.from_pandas_edgelist(val_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    test_df = pd.read_csv(test_path, sep="\t", header=None)
    test_df.columns = ["source", "edge", "target"]
    test_df = pd.DataFrame({"source": test_df.source, "target": test_df.target, "edge": test_df.edge})

    G_test = nx.from_pandas_edgelist(test_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    G = nx.compose_all([G_train, G_val, G_test])

    return G