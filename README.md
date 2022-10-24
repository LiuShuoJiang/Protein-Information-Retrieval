# Protein Structure Information Retrieval

## Description

Similar to [Foldseek](https://github.com/steineggerlab/foldseek), this project implements a protein structure database searching methodology, while the method used here is based on GVP-GNN for protein structure representation learning.

## Training Dataset

We use Foldseek to generate the ground-truth datasets.

### Query Database

We use [CATH/Gene3D dataset](https://www.cathdb.info/), see [this page](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/) to download the .pdb format dataset.

### Target Database

We use [Alphafold protein structure database](https://alphafold.ebi.ac.uk/), see [this page](https://alphafold.ebi.ac.uk/download#swissprot-section) to download the Swiss-Prot dataset (Huge!!! about 26GB compressed).

## App

The app will be constructed later.

## Package Requirements

[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)

[Biopython](https://biopython.org/)

[Biotite](https://github.com/biotite-dev/biotite)

[FAIR-ESM](https://github.com/facebookresearch/esm)

[Foldseek](https://github.com/steineggerlab/foldseek)

[Pandas](https://pandas.pydata.org/)

[WandB](https://wandb.ai/)
