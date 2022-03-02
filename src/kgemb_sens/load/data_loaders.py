# -*- coding: utf-8 -*-

"""For loading various KG datasets."""

import pandas as pd
import networkx as nx

# local data dir
##data_dir="/Users/dnsosa/.data/pykeen/datasets"
DATA_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"
PCNET_DIR = DATA_DIR


def load_benchmark_data_three_parts(dataset, data_dir=DATA_DIR):
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

    # NOTE: compose_all function was error-prone
    combined_edge_list = list(G_train.edges(data=True)) + list(G_val.edges(data=True)) + list(G_test.edges(data=True))

    G = nx.MultiDiGraph(combined_edge_list)

    return G


def load_drkg_data(dataset, data_dir=DATA_DIR, pcnet_filter=False, pcnet_dir=PCNET_DIR):

    # Assumes it's already extracted somewhere
    drkg_df = pd.read_csv(f"{data_dir}/PYKEEN_DATASETS/drkg.tsv", sep="\t")
    drkg_df.columns = ["source", "edge_data", "target"]

    print(f"Length of full DRKG is: {len(drkg_df)} edges.")

    drkg_df[['kg', 'edge', 'entity_types']] = drkg_df.edge_data.str.split('::', expand=True)

    if "gnbr" in dataset:
        dataset_name = "GNBR"
    elif "drugbank" in dataset:
        dataset_name = "DRUGBANK"
    elif "string" in dataset:
        dataset_name = "STRING"
    else:
        print("Not a valid dataset name!!")
        return None

    if "gg" in dataset:
        query_entity_types = "Gene:Gene"
    elif "drdz" in dataset:
        query_entity_types = "Compound:Disease"
    elif "drg" in dataset:
        query_entity_types = "Compound:Gene"
    else:
        print("Not a valid dataset name (unrecognized entity types)!!")
        return None

    filtered_df = drkg_df[(drkg_df.kg == dataset_name) & (drkg_df.entity_types == query_entity_types)].reset_index()
    print(f"Size of {dataset}: {len(filtered_df)} edges.")
    print(f"Found the following relationship types when filtering to {dataset}: {set(filtered_df.edge)}.")
    filtered_df = filtered_df[['source', 'edge', 'target']]

    if pcnet_filter and query_entity_types == "Gene:Gene":

        filtered_df['source_entrez'] = filtered_df.source.str.split('::', expand=True)[1]
        filtered_df['target_entrez'] = filtered_df.target.str.split('::', expand=True)[1]

        def generate_merge_id(x):
            sorted_id = sorted([x["source_entrez"], x["target_entrez"]])
            merge_id = "_".join([str(element) for element in sorted_id])
            return merge_id

        filtered_df["merge_id"] = filtered_df.apply(generate_merge_id, axis=1)
        # TODO: include the code here that reformats the pcnet file. Explain the cx to sif step.
        # "/Users/dnsosa/Downloads/pcnet_reformatted_df.csv"
        pcnet_df = pd.read_csv(f"{pcnet_dir}/pcnet_reformatted_df.csv")

        filtered_df = filtered_df.merge(pcnet_df, how='inner', on='merge_id', sort=True)[["source_entrez_x", "edge_x", "target_entrez_x"]]
        print(f"After including only pairs in PCnet--size of {dataset}: {len(filtered_df)} edges.")
        filtered_df.columns = ['source', 'edge', 'target']

    G = nx.from_pandas_edgelist(filtered_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    return G