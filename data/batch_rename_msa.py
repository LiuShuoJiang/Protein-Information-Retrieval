import os
import shutil
import Bio.SeqIO

folder = 'msa_output'

for file in os.listdir(folder):
    target_file = os.path.join(folder, file)
    record_name = ''
    for i, record in enumerate(Bio.SeqIO.parse(target_file, "fasta")):
        if i == 0:
            record_name = record.name
        break
    src = f'{folder}/{file}'
    dst = f'{folder}/{record_name}.a3m'
    shutil.copy(src, dst)

print('rename successful!')
