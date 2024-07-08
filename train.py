from tqdm import tqdm

import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import get_config, get_weights_file_path
from dataset.dataset import CandleDataset
from model.model import VAE
from einops import rearrange
import os


def train_model(config=None, save_only_prod_epoch=False):

    if config is None:
        config = get_config()

    train_data_dir = config['train_data_dir']
    val_data_dir = config['val_data_dir']
    seq_len = config['seq_len']
    d_model = config['d_model']
    batch_size = config['batch_size']
    N = config['N']
    dropout = config['dropout']
    d_ff = config['d_ff']
    epochs = config['epochs']
    lr = config['lr']
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    preload = config['preload']
    experiment_name = config['experiment_name']
    prediction_units = config['prediction_units']
    latent_dims = config['latent_dims']
    instrument = config['instrument']

    train_ds = CandleDataset(train_data_dir, seq_len, prediction_units, config)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = CandleDataset(val_data_dir, seq_len, prediction_units, config)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = VAE(latent_dims=latent_dims, seq_len=seq_len, d_model=d_model, N=N, d_ff=d_ff, dropout=dropout).to(device)

    # Tensorboard
    writer = SummaryWriter(experiment_name)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = torch.nn.MSELoss()


    initial_epoch = 0
    global_step = 0
    if preload is not None and not save_only_prod_epoch:
        model_filename = get_weights_file_path(model_folder, model_basename, preload)
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        model.load_state_dict(state['model_state_dict'])

    for epoch in range(initial_epoch, epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        n_iterations = len(train_dataloader)
        total_train_loss = 0
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len, d_model)
            proj_output = model(encoder_input)

            # calculate loss
            loss = loss_fn(proj_output, encoder_input)
            batch_iterator.set_postfix({'loss': loss.item()})

            total_train_loss += loss.item()

            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        train_loss_average = total_train_loss / n_iterations
        writer.add_scalar('train loss', train_loss_average, epoch)
        writer.flush()

        if not save_only_prod_epoch:
            # save model after each epoch
            if epoch % 2 == 0:
                model_filename = get_weights_file_path(model_folder, model_basename, f'{epoch:02d}')
                print(f"Saving model weights to {model_filename}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                }, model_filename)
                run_test(epoch, model, val_dataloader, device, seq_len, d_model)
                run_validation(epoch, model, val_dataloader, device, writer, loss_fn)
        else:
            if epoch == config['epochs'] - 1:
                model_filename = get_weights_file_path(model_folder, model_basename, f'{epoch:02d}')
                print(f"Saving model weights to {model_filename}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                }, model_filename)


def run_test(epoch, model, val_dataloader, device, seq_len, d_model):
    model.eval()
    with torch.no_grad():
        batch_iterator_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
        counter = 0
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            proj_output = model(encoder_input)
            first_candle_gt = encoder_input[0]
            first_candle_pred = proj_output[0]
            # reshape to (500, 4)
            first_candle_gt = rearrange(first_candle_gt, '(seq_len d_model) -> seq_len d_model', seq_len=seq_len, d_model=d_model)
            first_candle_pred = rearrange(first_candle_pred, '(seq_len d_model) -> seq_len d_model', seq_len=seq_len, d_model=d_model)
            # convert to cpu and numpy
            first_candle_gt = first_candle_gt.cpu().numpy()
            first_candle_pred = first_candle_pred.cpu().numpy()
            plot_candle_prediction(first_candle_gt, first_candle_pred, epoch, counter)
            counter += 1
            if counter > 4:
                break


import matplotlib.pyplot as plt

def plot_candle_prediction(gt_candles, pred_candles, epoch, counter):
    # gt_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # pred_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # I want to plot the high, low and mid of gt_candles as a green line where high and low lines are dashed
    # I want to plot the high, low and mid of pred_candles as a red line where high and low lines are dashed
    # Finally save the plot to a file
    fig, ax = plt.subplots()
    # ax.plot(gt_candles[:, 0], 'g-', label='gt open')
    ax.plot(gt_candles[:, 1], 'b-', label='gt high')
    ax.plot(gt_candles[:, 2], 'b--', label='gt low')
    # ax.plot(gt_candles[:, 3], 'g--', label='gt close')
    # gt_mid = (gt_candles[:, 1] + gt_candles[:, 2]) / 2
    # ax.plot(gt_mid, 'g-', label='gt mid')
    # ax.plot(pred_candles[:, 0], 'r-', label='pred open')
    ax.plot(pred_candles[:, 1], 'r-', label='pred high')
    ax.plot(pred_candles[:, 2], 'r--', label='pred low')
    # ax.plot(pred_candles[:, 3], 'r--', label='pred close')
    # pred_mid = (pred_candles[:, 1] + pred_candles[:, 2]) / 2
    # ax.plot(pred_mid, 'r--', label='pred mid')
    plt.legend()
    # save plot
    save_folder = f'figures/{counter}'
    # create folder if it does not exist and delete all files in it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.savefig(f'figures/{counter}/candle_prediction_{epoch}.png')


def run_validation(epoch, model, val_dataloader, device, writer, loss_fn):
    model.eval()
    with torch.no_grad():
        batch_iterator_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
        total_val_loss = 0
        n_iterations = len(val_dataloader)
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            proj_output = model(encoder_input)
            val_loss = loss_fn(proj_output, encoder_input)
            total_val_loss += val_loss.item()
            batch_iterator_val.set_postfix({'val_loss': val_loss.item()})

        val_loss_average = total_val_loss / n_iterations
        writer.add_scalar('val loss', val_loss_average, epoch)
        writer.flush()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model(config=None)