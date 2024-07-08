from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config.config import get_config, get_weights_file_path
from dataset.dataset import CandleDataset
from model.model import VAE
import pickle


def create_db(config=None, use_prod_model=False):
    if config is None:
        config = get_config()
    train_data_dir = config['train_data_dir']
    seq_len = config['seq_len']
    d_model = config['d_model']
    batch_size = config['batch_size']
    N = config['N']
    d_ff = config['d_ff']
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    prediction_units = config['prediction_units']
    latent_dims = config['latent_dims']

    if use_prod_model:
        preload = config['epochs'] - 1
    else:
        preload = config['preload']

    train_ds = CandleDataset(train_data_dir, seq_len, prediction_units, config)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = VAE(latent_dims=latent_dims, seq_len=seq_len, d_model=d_model, N=N, d_ff=d_ff, dropout=0).to(device)

    model_filename = get_weights_file_path(model_folder, model_basename, preload)
    print(f"Loading model weights from {model_filename}")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    batch_iterator = tqdm(train_dataloader, desc=f"Creating db")
    n_iterations = len(train_dataloader)

    # data base consists pairs of latent vectors and labels
    data_base = []

    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len, d_model)
        proj_output = model.encoder(encoder_input)
        # proj_output to cpu
        proj_output = proj_output.cpu().detach().numpy()

        label = batch['label']
        label = label.cpu().detach().numpy()

        for i in range(len(proj_output)):
            data_base.append((proj_output[i], label[i]))


    # save data base to file
    database_file_name = f"{config['data_base_dir']}{config['instrument']}.pkl"
    with open(database_file_name, 'wb') as f:
        pickle.dump(data_base, f)


if __name__ == '__main__':
    create_db(config=None, use_prod_model=False)