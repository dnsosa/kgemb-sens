# -*- coding: utf-8 -*-

"""For loading various KG datasets."""

import pandas as pd
import networkx as nx

from kgemb_sens.transform.graph_utilities import undirect_multidigraph


# local data dir
##data_dir="/Users/dnsosa/.data/pykeen/datasets"
DATA_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"
PCNET_DIR = DATA_DIR
COVIDKG_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/covid19kg"


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


def load_drkg_data(dataset, data_dir=DATA_DIR, pcnet_filter=False, pcnet_dir=PCNET_DIR, dengue_filter=False, dengue_expand_depth=1):

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
    elif "hetionet" in dataset:
        dataset_name = "Hetionet"
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
        query_entity_types = None
        print(f"No entity query types specified, just returning all of {dataset}")

    # Filter the dataset from DRKG
    if query_entity_types is None:
        filtered_df = drkg_df[drkg_df.kg == dataset_name].reset_index()
    else:
        filtered_df = drkg_df[(drkg_df.kg == dataset_name) & (drkg_df.entity_types == query_entity_types)].reset_index()

    # Filter only the relevant relation types between Dr, Dz, and G
    if dataset == "gnbr":
        filtered_df = filtered_df[(filtered_df.edge != "in_tax")]
        print(f"Filtering out 'in_tax' edges for GNBR KG.")
    elif dataset == "hetionet":
        hetionet_drdzg_edges = {"DaG", "DdG", "DuG", "GcG", "GiG", "Gr>G", "CtD", "CpD", "CuG", "CdG", "CbG"}
        filtered_df = filtered_df[filtered_df.edge.isin(hetionet_drdzg_edges)]
        print(f"Filtering in only {hetionet_drdzg_edges} edge types in Hetionet.")

    # Summarize what happened
    print(f"Size of {dataset}: {len(filtered_df)} edges.")
    print(f"Found the following relationship types when filtering to {dataset}: {set(filtered_df.edge)}.")
    filtered_df = filtered_df[['source', 'edge', 'target']]

    # Extra processing steps for working with PPIs, for instance for filtering based on PCNet.
    # TODO: Apply PCNet filtration to the gene-gene portion ONLY even if using the whole KG?
    if query_entity_types == "Gene:Gene":
        filtered_df['source_entrez'] = filtered_df.source.str.split('::', expand=True)[1]
        filtered_df['target_entrez'] = filtered_df.target.str.split('::', expand=True)[1]

        if pcnet_filter:
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

        else:
            filtered_df = filtered_df[["source_entrez", "edge", "target_entrez"]]

        filtered_df.columns = ['source', 'edge', 'target']

    G = nx.from_pandas_edgelist(filtered_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    if ("gg" in dataset) and dengue_filter:

        dengue_genes = {'10417', '1107', '1524', '1669', '256987', '259197', '26289', '2634', '2992', '51161', '51225',
                        '55799', '55809', '57156', '5797', '59084', '64222', '9254', '9402', '9580'}

        G = undirect_multidigraph(G)
        dengue_nodes_G = set(dengue_genes).intersection(set(G.nodes()))
        # print(f"Dengue nodes: {dengue_nodes_G}")

        dn_edges = []
        for dengue_node in dengue_nodes_G:
            visited_edges = list(nx.bfs_edges(G, source=dengue_node, depth_limit=dengue_expand_depth))

            visited_edges_attrs = []
            for e in visited_edges:
                rel_data = G.get_edge_data(*e)
                edge_rels = set([rel_data[k]['edge'] for k in rel_data.keys()])
                for rel in edge_rels:
                    visited_edges_attrs.append((e[0], e[1], rel))

            dn_edges += visited_edges_attrs

        dn_edges = list(set(dn_edges))
        dn_edges = [(u, v, {'edge': r}) for u, v, r in dn_edges]
        G_dengue = nx.MultiGraph()  # Note NOT directed
        G_dengue.add_edges_from(dn_edges)

        return G_dengue

    return G


def load_covid_graph(data_dir=COVIDKG_DIR):
    covid_path = f"{data_dir}/_cache.csv"
    covid_df = pd.read_csv(covid_path, header=None, sep="\t")
    covid_df.columns = ["source", "edge", "target", "edge_attributes"]

    G_covid = nx.from_pandas_edgelist(covid_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    return G_covid
