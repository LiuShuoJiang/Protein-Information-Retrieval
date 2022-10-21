# A Detailed Instruction on How to Prepare for the Datasets

All the following operations are based on the data folder.

## Download and Set up Datasets

1. Download the file "cath-dataset-nonredundant-S20.pdb.tgz" from [this link](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S20.pdb.tgz) and extract it ***twice*** to the current folder. The unpacked folder should be called "dompdb" and should be around 1.15GB in size.

2. Run "batch_rename_pdb.py" to rename all the files in "dompdb" directory with the `.pdb` suffix.

3. Download the file "swissprot_pdb_v2.tar" from [this link](https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/swissprot_pdb_v2.tar) (about 26GB) and extract it *once* to the current folder. The unpacked folder should be called "swissprot_pdb_v2".

4. Open the terminal, go to "swissprot_pdb_v2" directory (using `cd` command) and run command `for g in *.gz; do gunzip $g; done`. (Warning: It may take 3~5 hours to finish this command and the unpacked folder is about 120GB in size.)

## Generate Ground Truth MSAs using Foldseek

1. Install **Foldseek** by following the instructions from its [official website](https://github.com/steineggerlab/foldseek).

2. Run the following commands: (Warning: It may take about an hour to finish all these commands.)

```bash
foldseek createdb dompdb/ QueryDatabase/QueryDB
foldseek createdb swissprot_pdb_v2/ TargetDatabase/TargetDB
foldseek search QueryDatabase/QueryDB TargetDatabase/TargetDB aln tmpFolder -a
foldseek result2msa QueryDatabase/QueryDB TargetDatabase/TargetDB aln msa --msa-format-mode 6
foldseek unpackdb msa msa_output --unpack-suffix a3m --unpack-name-mode 0
```

3. Run "batch_rename_msa.py" to rename all files in the "msa_output" folder.

4. (Optional) Run "generate_msa_index.py" to generate the training and testing index files, and move the generate `.csv` files to the "split" directory from the project root.

5. Copy all `.pdb` files in "dompdb" folder into the "swissprot_pdb_v2" folder. You may use the `cp -a ./dompdb/. /swissprot_pdb_v2/` command.
