import os
import Bio.SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split

msa_all = os.listdir('msa_output')
names = []
num_of_msa = []

# ===================================generate MSA index======================================
for i, msa in enumerate(msa_all):
    target_dir = os.path.join('msa_output', msa)
    count = 0
    file_name = ''
    for record in Bio.SeqIO.parse(target_dir, "fasta"):
        if count == 0:
            file_name = record.name
        count += 1
        if count >= 32:
            break
    if count >= 32:
        names.append(file_name)
        num_of_msa.append(count)
    if i % 500 == 0:
        print(f'processed {i} MSAs')

data = pd.DataFrame({'protein': names, 'msas': num_of_msa})
data.to_csv('msa_all_index.csv', header=False, index=False)
print('index saved!')

# =======================generate training, testing and validation data index================
X, y = data['protein'], data['msas']
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

# train is now 70% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),
                                                random_state=42)

train_concat = pd.concat([x_train, y_train], axis=1)
train_concat.to_csv('train_split.csv', header=False, index=False)

test_concat = pd.concat([x_test, y_test], axis=1)
test_concat.to_csv('test_split.csv', header=False, index=False)

val_concat = pd.concat([x_val, y_val], axis=1)
val_concat.to_csv('val_split.csv', header=False, index=False)

print('All train, test and validation index saved!')
