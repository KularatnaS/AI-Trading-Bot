import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from config.config import get_config, get_weights_file_path
from dataset.dataset import CandleDataset
from model.model import ClassificationModel

import matplotlib.pyplot as plt
import os

def plot_candle_prediction(gt_candles, pred_candles, decision, counter, status, profit_made, ATR):
    # gt_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # pred_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # I want to plot the high, low and mid of gt_candles as a green line where high and low lines are dashed
    # I want to plot the high, low and mid of pred_candles as a red line where high and low lines are dashed
    # Finally save the plot to a file
    fig, ax = plt.subplots()

    ax.plot(pred_candles[:, 1], 'r-', label='future high')
    ax.plot(pred_candles[:, 2], 'r--', label='future low')

    ax.plot(gt_candles[:, 1], 'b-', label='past high')
    ax.plot(gt_candles[:, 2], 'b--', label='past low')
    ax.plot(gt_candles[:, 3], 'g+', label='past closing price')

    # plot closing price - 2 * ATR and closing price + 2 * ATR two horizontal lines
    ax.axhline(y=gt_candles[-1, 3] + 2 * ATR, color='y', linestyle='-', label='2 * ATR')
    ax.axhline(y=gt_candles[-1, 3] - 2 * ATR, color='y', linestyle='-', label='-2 * ATR')

    plt.legend()

    if decision == 0:
        decision = "buy"
    elif decision == 1:
        decision = "sell"

    plt.title(f"Decision: {decision}, Profit made: {profit_made}")
    # save folder
    save_folder = "predictions"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if profit_made:
        plt.savefig(f'{save_folder}/{counter}_{decision}_profit.png')
    else:
        plt.savefig(f'{save_folder}/{counter}_{decision}_loss.png')

config = get_config()
data_dir = config['data_dir']
train_data_dir = config['train_data_dir']
val_data_dir = config['val_data_dir']
test_data_dir = config['test_data_dir']
seq_len = config['seq_len']
d_model = config['d_model']
batch_size = config['batch_size']
N = config['N']
h = config['h']
dropout = config['dropout']
d_ff = config['d_ff']
epochs = config['epochs']
lr = config['lr']
model_folder = config['model_folder']
model_basename = config['model_basename']
preload = config['preload']
experiment_name = config['experiment_name']
data_dir = config['data_dir']
n_trade_decisions = config['n_trade_decisions']
prediction_units = config['prediction_units']
latent_dims = config['latent_dims']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model = ClassificationModel(n_trade_decisions=n_trade_decisions, seq_len=seq_len, d_model=d_model, N=N, d_ff=d_ff, dropout=0).to(device)

model_filename = get_weights_file_path(model_folder, model_basename, preload)
print(f"Loading model weights from {model_filename}")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

test_ds = CandleDataset(test_data_dir, seq_len, prediction_units)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

val_ds = CandleDataset(val_data_dir, seq_len, prediction_units)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model.eval()

batch_iterator = tqdm(test_dataloader, desc=f"Creating db")



def get_predicted_label(classification_output, trust_threshold=0.5):
    # classification_output is a numpy array of shape (n_trade_decisions,)
    # returns the index of the maximum value in classification_output if the maximum value is greater than trust_threshold
    # otherwise returns None
    classification_prob = np.exp(classification_output) / np.sum(np.exp(classification_output))
    if np.max(classification_prob) > trust_threshold:
        return np.argmax(classification_prob)
    else:
        return None


def made_a_profit(predicted_label, future_candles, past_candles):
    # use first future candle get start price
    buy_or_sell_price = future_candles[0][0]

    # use last 14 past candles to calculate average true range
    high = past_candles[-14:, 1]
    low = past_candles[-14:, 2]

    # calculate average true range
    ranges = high - low
    average_true_range = np.mean(ranges)

    # convert future candles to numpy
    future_candles = np.array(future_candles)
    # use future candles to get max and min and there indices
    max_price, max_index = np.max(future_candles[:, 1]), np.argmax(future_candles[:, 1])
    min_price, min_index = np.min(future_candles[:, 2]), np.argmin(future_candles[:, 2])

    final_price = future_candles[-1, 3]
    exit_price = None

    made_a_profit_out = False
    # the bid is with a reward to risk ratio of 1.5 using the average true range
    atr_exit_factor = config['atr_exit_factor']
    if predicted_label == 0:  # buy
        if ((max_price - buy_or_sell_price) >= (atr_exit_factor * average_true_range)):
            if min_price >= buy_or_sell_price:
                made_a_profit_out = True
                exit_price = buy_or_sell_price + (atr_exit_factor * average_true_range)
            elif min_price < buy_or_sell_price:
                if ((buy_or_sell_price - min_price) < (atr_exit_factor * average_true_range)):
                    made_a_profit_out = True
                    exit_price = buy_or_sell_price + (atr_exit_factor * average_true_range)
                else:
                    if max_index < min_index:
                        made_a_profit_out = True
                        exit_price = buy_or_sell_price + (atr_exit_factor * average_true_range)
                    else:
                        made_a_profit_out = False
                        exit_price = buy_or_sell_price - (atr_exit_factor * average_true_range)
        else:
            if min_price >= (buy_or_sell_price - (atr_exit_factor * average_true_range)):
                if (final_price - buy_or_sell_price) > config['spread']:
                    made_a_profit_out = True
                    exit_price = final_price
                else:
                    exit_price = final_price
                    made_a_profit_out = False
            else:
                made_a_profit_out = False
                exit_price = buy_or_sell_price - (atr_exit_factor * average_true_range)

    elif predicted_label == 1:  # sell
        if ((buy_or_sell_price - min_price) >= (atr_exit_factor * average_true_range)):
            if max_price <= buy_or_sell_price:
                made_a_profit_out = True
                exit_price = buy_or_sell_price - (atr_exit_factor * average_true_range)
            elif max_price > buy_or_sell_price:
                if ((max_price - buy_or_sell_price) < (atr_exit_factor * average_true_range)):
                    made_a_profit_out = True
                    exit_price = buy_or_sell_price - (atr_exit_factor * average_true_range)
                else:
                    if min_index < max_index:
                        made_a_profit_out = True
                        exit_price = buy_or_sell_price - (atr_exit_factor * average_true_range)
                    else:
                        made_a_profit_out = False
                        exit_price = buy_or_sell_price + (atr_exit_factor * average_true_range)
        else:
            if max_price <= (buy_or_sell_price + (atr_exit_factor * average_true_range)):
                if (buy_or_sell_price - final_price) > config['spread']:
                    made_a_profit_out = True
                    exit_price = final_price
                else:
                    exit_price = final_price
                    made_a_profit_out = False
            else:
                made_a_profit_out = False
                exit_price = buy_or_sell_price + (atr_exit_factor * average_true_range)

    if made_a_profit_out:
        profit = np.abs(exit_price - buy_or_sell_price) * config['leverage'] - config['spread'] * config['leverage']
    else:
        profit = -np.abs(exit_price - buy_or_sell_price) * config['leverage'] - config['spread'] * config['leverage']

    return made_a_profit_out, average_true_range, profit



number_of_correct_predictions = 0
n_profitable_trades = 0
total_profit = 0
total_predictions = 0

for batch in batch_iterator:
    encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len, d_model)
    original_encoder_input = batch['original_encoder_input']
    proj_output = model.encoder(encoder_input)
    # proj_output to cpu
    proj_output = proj_output.cpu().detach().numpy()

    label = batch['label']
    future_candles = batch['future_candles']
    gt_label = label.cpu().detach().numpy()

    for i in range(len(proj_output)):
        # encode_input to cpu
        input_i = original_encoder_input[i]
        # reshape input_i to (seq_len, d_model)
        input_i = input_i.reshape(seq_len, d_model)
        input_i = input_i.cpu().detach().numpy()

        # concatenate input_i and future_candles[i]
        input_with_future = np.concatenate((input_i, future_candles[i]), axis=0)

        latent_vector = proj_output[i]
        predicted_label = get_predicted_label(proj_output[i])
        if predicted_label is not None:
            if predicted_label in [0, 1]:  # only consider buy and sell
                profit_made, ATR, profit = made_a_profit(predicted_label, future_candles[i], input_i)
                total_profit += profit
                if profit_made:
                    n_profitable_trades += 1
                if predicted_label == gt_label[i]:
                    number_of_correct_predictions += 1
                    plot_candle_prediction(input_i, input_with_future, predicted_label, total_predictions, 'correct', profit_made, ATR)
                else:
                    plot_candle_prediction(input_i, input_with_future, predicted_label, total_predictions, 'incorrect', profit_made, ATR)
                total_predictions += 1
                print(f"Current accuracy: {number_of_correct_predictions / total_predictions}, total predictions: {total_predictions}")
                print(f"profitable trades: {n_profitable_trades / total_predictions}")
                print(f"total profit: {total_profit}")


print(f"Accuracy: {number_of_correct_predictions / total_predictions}")


