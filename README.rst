kgemb-sens
=============
A package for setting up and running experiments on the sensitivity of knowledge graph (KG) embedding methods to
sparsity and inconsistency.

This package was written as source code to accompany the work `Characterizing the Effect of Sparsity and Inconsistency
on Knowledge Graph Link Prediction` [CURRENTLY IN PROGRESS]. This package:

- Loads various KG embedding datasets
- Automates experiments on inducing sparsity in a KG with various tunable parameters
- Automates experiments on inducing contradictory information in a KG with various tunable parameters
- Runs PyKEEN-based embedding pipeline
- Provides code for plotting small example networks
- Tests functionality on toy network

Ignore the Junk
---------------
This repository uses a ``.gitignore`` file to make sure no junk gets committed. GitHub will ask you if
you want a pre-populated ``.gitignore`` added to your repo on creation. You can also go to https://www.gitignore.io/
to get more options.

Things that are especially bad to commit to repos:

- compiled python files (*.pyc)
- Jupyter notebook checkpoint folders (.ipynb_checkpoints/)
- documentation builds (let ReadTheDocs take care of this!)
- tox and other automation/build tool caches
- basically any file you didn't make on purpose

Usage
-----

TODO