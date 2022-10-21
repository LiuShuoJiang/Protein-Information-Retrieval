import os

folder = 'dompdb'

# add .pdb suffix for the original files
for file in os.listdir(folder):
    src = f'{folder}/{file}'
    dst = f'{folder}/{file}.pdb'
    os.rename(src, dst)
