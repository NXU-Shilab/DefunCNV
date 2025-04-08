import random

import torch
import h5py
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from Bio import SeqIO

import os, sys
model_path = os.path.abspath(os.path.join('/mnt/data0/users/baiy/CNV/code/model/'))
sys.path.append(model_path)
from train_predict.cnn_with_swin import Sei_2d_with_Swin

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2024)

def read_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append(str(record.seq))

    return sequences


def sequences_one_hot(sequences):
    # one-hot
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    bases = ['A', 'C', 'G', 'T']
    num_classes = 4  # A, C, G, T
    data = []

    for seq in sequences:
        seq_encoded = []
        for base in seq:
            if base in mapping:
                seq_encoded.append(mapping[base])
            else:
                # Randomly choose one (A, C, G, T) to replace N
                random_base = np.random.choice(bases)
                seq_encoded.append(mapping[random_base])
        seq_one_hot = np.eye(num_classes)[seq_encoded]
        data.append(seq_one_hot)
        

    data = np.array(data)
    data = data.transpose(0, 2, 1)
    return data


def save_predictions(predictions, output_path):
    with h5py.File(output_path, 'w') as h5_file:
        h5_file.create_dataset('predictions', data=predictions)


def main():
    model_path = "/mnt/data0/users/baiy/CNV/code/model/best_model/best_model.pth.tar"
    data_path = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_alt_sequence.fa"
    output_path = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_alt_sequence_predictions.h5"
    batch_size = 1

    model = Sei_2d_with_Swin()
    device_ids = [6]
    device = torch.device("cuda:%s" % device_ids[0] if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    state_dict = torch.load(model_path)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()

    if len(model_keys) != len(state_dict_keys):
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ValueError("Loaded state dict does not match the model "
                             "architecture specified - please check that you are "
                             "using the correct architecture file and parameters.\n\n"
                             "{0}".format(e))

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = state_dict[k2]
        try:
            new_state_dict[k1] = value
        except Exception as e:
            raise ValueError(
                "Failed to load weight from module {0} in model weights "
                "into model architecture module {1}. (If module name has "
                "an additional prefix `model.` it is because the model is "
                "wrapped in `selene_sdk.utils.NonStrandSpecific`. This "
                "error was raised because the underlying module does "
                "not match that expected by the loaded model:\n"
                "{2}".format(k2, k1, e))
    model.load_state_dict(new_state_dict)

    model.eval()

    # Read and preprocess sequence data
    sequences = read_fasta(data_path)
    print(len(sequences))
    data = sequences_one_hot(sequences)
    print(data.shape)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    
    print("Start prediction")

    with torch.no_grad(): 
        for i, batch in enumerate(data_loader):
            batch_data = batch[0].to(device)
            predictions = model(batch_data)
            all_predictions.append(predictions.cpu().numpy())
            print(f"Batch {i + 1}/{len(data_loader)} processed")

    all_predictions = np.vstack(all_predictions)
    
    # Save prediction results to.H5 file
    save_predictions(all_predictions, output_path)

    print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    main()
