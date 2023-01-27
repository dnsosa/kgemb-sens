# -*- coding: utf-8 -*-

"""Visualizations of the experimental results across multiple network perturbation settings."""

import numpy as np
import matplotlib.pyplot as plt


def plot_degree_distribution(G):
    """
    Plot the degree distribution of nodes in G.

    :param G: input KG
    """
    pass


def coarse_corrupt_performance_scatter(df):
    """
    Plot the performance of models in different total corruption settings.

    :param df: dataframe of compiled results across all conditions
    """
    pass


def coarse_corrupt_performance_drop(df):
    """
    Plot the drop in performance in different total corruption settings.

    :param df: dataframe of compiled results across all conditions
    """
    pass


def fine_corrupt_performance(df):
    """
    Plot the performance of different groups in fine-grained corruption settings.

    :param df: dataframe of compiled results across all conditions
    """
    pass
