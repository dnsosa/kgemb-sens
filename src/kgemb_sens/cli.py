# -*- coding: utf-8 -*-

"""Command line interface for kgemb_sens."""

import datetime
import os
import shutil
from random import randrange

import click
import pandas as pd

from .analyze import *
from .transform import *
from .load import *


# TODO Fill this all in
@click.command()
@click.option('--train/--no-train', 'train', default=False)
@click.option('--output_folder', 'output_dir')
@click.option('--roberta/--no-roberta', 'roberta', default=True)
@click.option('--logistic-regression/--no-logistic-regression', 'logistic_model', default=True)
def main(extract, train, bluebert_train, bluebert_model_path, report, bluebert_report, multi_class, cord_version,
         sbert, logistic_model):
    """Run main function."""
    # Model parameters
    pass


if __name__ == '__main__':
    main()
