import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler, Adam
import esm
from GVP_Model import MySimpleRepresentation
from utils import fetch_file_names
from data import MyProteinDataset, MyBatchConverter
from training import train
import wandb
import warnings
import os
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.filterwarnings("ignore")

# path = 'autodl-tmp/msa_output/'
path = 'data/msa_output/'

train_batch_size = 4
validation_batch_size = 2
lr = 1e-4
epochs = 50
evaluation_per_step = 250
run_device = 'cuda'


def init_wandb():
    wandb.init(
        project="Protein-Information-Retrieval",
        config={
            "optim": "AdamW",
            "lr": lr,
            "train_batch_size": train_batch_size,
            "evaluation_per_step": evaluation_per_step,
        },
        settings=wandb.Settings(start_method="fork")
    )


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    gc.collect()
    torch.cuda.empty_cache()
    _, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    print('initial model loaded!')

    device = torch.device(run_device)
    # device = torch.device('cpu')

    model = MySimpleRepresentation()
    model.load_state_dict(torch.load('autodl-tmp/saved_model/99.pth'))
    model.to(device=device)

    count = 0

    init_wandb()

    train_path = 'split/train_split.csv'
    train_names, train_lines = fetch_file_names(train_path)
    train_names = [path + name + '.a3m' for name in train_names]

    val_path = 'split/val_split.csv'
    val_names, val_lines = fetch_file_names(val_path)
    val_names = [path + name + '.a3m' for name in val_names]

    training_set = MyProteinDataset(train_names, train_lines)
    validation_set = MyProteinDataset(val_names, val_lines)

    train_batch = training_set.get_batch_indices(train_batch_size)
    val_batch = validation_set.get_batch_indices(validation_batch_size)

    batch_converter = MyBatchConverter(alphabet)

    train_loader = DataLoader(dataset=training_set, collate_fn=batch_converter, batch_sampler=train_batch,
                              num_workers=12)
    val_loader = DataLoader(dataset=validation_set, collate_fn=batch_converter, batch_sampler=val_batch, num_workers=12)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.85)

    train(model, train_loader, val_loader, epochs, optimizer, evaluation_per_step, acc_step=1)
