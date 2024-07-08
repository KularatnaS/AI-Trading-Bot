import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from config.config import get_config, get_weights_file_path
from dataset.dataset import CandleDataset
from model.model import VAE
from sklearn.neighbors import NearestNeighbors
from dataset.dataset import calculate_atr

import matplotlib.pyplot as plt
import os

def get_model(config, device, use_prod_model):
    model = VAE(latent_dims=config["latent_dims"], seq_len=config["seq_len"], d_model=config["d_model"], N=config["N"],
                d_ff=config["d_ff"], dropout=0).to(device)

    if use_prod_model:
        preload = config['epochs'] - 1
    else:
        preload = config['preload']

    model_filename = get_weights_file_path(config["model_folder"], config["model_basename"], preload)
    if device == "cpu":
        state = torch.load(model_filename, map_location=torch.device('cpu'))
    else:
        state = torch.load(model_filename)

    model.load_state_dict(state['model_state_dict'])
    return model
def get_nn_map_and_labels_db(config):
    database_file_name = config['data_base_dir'] + f'{config["instrument"]}.pkl'
    with open(database_file_name, 'rb') as f:
        data_base = pickle.load(f)

    latent_vectors_db = [x[0] for x in data_base]
    labels_db = [x[1] for x in data_base]

    labels_db = np.array(labels_db)
    latent_vectors_db = np.array(latent_vectors_db)

    # fit a nearest neighbor model to latent vectors
    nn_map = NearestNeighbors(n_neighbors=int(config['seq_len']), algorithm='auto').fit(latent_vectors_db)
    return nn_map, labels_db

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


def find_label_of_closest_latent_vector(nn_map, labels_db, latent_vector, config):
    # find the lables of the closest latent vectors
    distances, indices = nn_map.kneighbors([latent_vector])
    indices = indices.flatten()
    # closest_labels = labels_db[indices]

    # return closes label
    distances = distances.flatten()
    index_of_closest_label = np.argmin(distances)
    return labels_db[indices[index_of_closest_label]]

    # unique, counts = np.unique(closest_labels, return_counts=True)
    # n_buy_counts = len(np.argwhere(closest_labels == 0))
    # n_sell_counts = len(np.argwhere(closest_labels == 1))
    # n_hold_counts = len(np.argwhere(closest_labels == 2))
    #
    # n_hold_percentage = n_hold_counts / (n_buy_counts + n_sell_counts + n_hold_counts)
    #
    # if n_hold_percentage < config['hold_min_prob']:
    #
    #     percentage_of_buy = n_buy_counts / (n_buy_counts + n_sell_counts)
    #     percentage_of_sell = n_sell_counts / (n_buy_counts + n_sell_counts)
    #     if percentage_of_buy > config['buy_sell_decision_min_prob']:
    #         return 0
    #     elif percentage_of_sell > config['buy_sell_decision_min_prob']:
    #         return 1
    #     else:
    #         return None
    # else:
    #     return None

    # label with the most counts
    # label_with_most_counts = unique[np.argmax(counts)]
    # print(label_with_most_counts)
    # return label_with_most_counts

    # # check percentage of the label
    # percentage = np.max(counts) / np.sum(counts)
    # if percentage > 1/2:
    #     return label_with_most_counts
    # else:
    #     return None


def made_a_profit(predicted_label, future_candles, past_candles, config, average_true_range):
    # use first future candle get start price
    buy_or_sell_price = future_candles[0][0]

    # convert future candles to numpy
    future_candles = np.array(future_candles)
    # use future candles to get max and min and there indices
    max_price, max_index = np.max(future_candles[:, 1]), np.argmax(future_candles[:, 1])
    min_price, min_index = np.min(future_candles[:, 2]), np.argmin(future_candles[:, 2])

    final_price = future_candles[-1, 3]
    exit_price = None

    made_a_profit_out = False
    trade_cancelled = False
    trade_loss_with_hitting_stop_loss = False
    trade_profit_with_hitting_take_profit = False
    # the bid is with a reward to risk ratio of 1.5 using the average true range
    atr_exit_factor_profit = config['atr_exit_factor_profit']
    atr_exit_factor_loss = config['atr_exit_factor_loss']

    delta_profit = atr_exit_factor_profit * average_true_range + config['spread']
    delta_loss = atr_exit_factor_loss * average_true_range - config['spread']

    if predicted_label == 0:  # buy
        if ((max_price - buy_or_sell_price) >= (delta_profit)):
            if min_price >= buy_or_sell_price:
                made_a_profit_out = True
                exit_price = buy_or_sell_price + (delta_profit)
                trade_profit_with_hitting_take_profit = True
            elif min_price < buy_or_sell_price:
                if ((buy_or_sell_price - min_price) < (delta_loss)):
                    made_a_profit_out = True
                    exit_price = buy_or_sell_price + (delta_profit)
                    trade_profit_with_hitting_take_profit = True
                else:
                    if max_index < min_index:
                        made_a_profit_out = True
                        exit_price = buy_or_sell_price + (delta_profit)
                        trade_profit_with_hitting_take_profit = True
                    else:
                        made_a_profit_out = False
                        exit_price = buy_or_sell_price - (delta_loss)
                        trade_loss_with_hitting_stop_loss = True
        else:
            trade_cancelled = True
            if min_price >= (buy_or_sell_price - (delta_loss)):
                if (final_price - buy_or_sell_price) > config['spread']:
                    made_a_profit_out = True
                    exit_price = final_price
                else:
                    exit_price = final_price
                    made_a_profit_out = False
            else:
                made_a_profit_out = False
                exit_price = buy_or_sell_price - (delta_loss)

    elif predicted_label == 1:  # sell
        if ((buy_or_sell_price - min_price) >= (delta_profit)):
            if max_price <= buy_or_sell_price:
                made_a_profit_out = True
                exit_price = buy_or_sell_price - (delta_profit)
                trade_profit_with_hitting_take_profit = True
            elif max_price > buy_or_sell_price:
                if ((max_price - buy_or_sell_price) < (delta_loss)):
                    made_a_profit_out = True
                    exit_price = buy_or_sell_price - (delta_profit)
                    trade_profit_with_hitting_take_profit = True
                else:
                    if min_index < max_index:
                        made_a_profit_out = True
                        exit_price = buy_or_sell_price - (delta_profit)
                        trade_profit_with_hitting_take_profit = True
                    else:
                        made_a_profit_out = False
                        exit_price = buy_or_sell_price + (delta_loss)
                        trade_loss_with_hitting_stop_loss = True
        else:
            trade_cancelled = True
            if max_price <= (buy_or_sell_price + (delta_loss)):
                if (buy_or_sell_price - final_price) > config['spread']:
                    made_a_profit_out = True
                    exit_price = final_price
                else:
                    exit_price = final_price
                    made_a_profit_out = False
            else:
                made_a_profit_out = False
                exit_price = buy_or_sell_price + (delta_loss)

    if made_a_profit_out:
        profit = np.abs(exit_price - buy_or_sell_price) - config['spread']
    else:
        profit = -np.abs(exit_price - buy_or_sell_price) - config['spread']

    # print(f"average true range: {average_true_range}, delta profit: {delta_profit}, delta loss: {delta_loss}, "
    #       f"made a profit: {made_a_profit_out}, profit: {profit}")

    return made_a_profit_out, profit, trade_cancelled, trade_loss_with_hitting_stop_loss, trade_profit_with_hitting_take_profit


def run_test(config=None, use_prod_model=False):

    if config is None:
        config = get_config()

    test_data_dir = config['test_data_dir']
    seq_len = config['seq_len']
    d_model = config['d_model']
    batch_size = config['batch_size']
    prediction_units = config['prediction_units']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device {device}")
    model = get_model(config, device, use_prod_model)
    model.eval()

    test_ds = CandleDataset(test_data_dir, seq_len, prediction_units, config)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    #batch_iterator = tqdm(test_dataloader, desc=f"Testing model")
    nn_map, labels_db = get_nn_map_and_labels_db(config)

    number_of_correct_predictions = 0
    n_profitable_trades = 0
    total_profit = 0
    total_predictions = 0
    total_trade_opportunities = 0
    n_cancelled_trades = 0
    n_trade_loss_with_hitting_stop_loss = 0
    n_trade_profit_with_hitting_take_profit = 0

    hold_counter = 0
    trade = True
    trade_made = False
    profit_made = True

    profit_only = 0
    loss_only = 0

    for n in range(len(test_dataloader)):
        batch = next(iter(test_dataloader))
        encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len, d_model)
        original_encoder_input = batch['original_encoder_input']
        proj_output = model.encoder(encoder_input)
        # proj_output to cpu
        proj_output = proj_output.cpu().detach().numpy()

        label = batch['label']
        future_candles = batch['future_candles']
        gt_label = label.cpu().detach().numpy()

        for i in range(len(proj_output)):
            total_trade_opportunities += 1
            if trade:
                # encode_input to cpu
                input_i = original_encoder_input[i]
                # reshape input_i to (seq_len, d_model)
                input_i = input_i.reshape(seq_len, d_model)
                input_i = input_i.cpu().detach().numpy()

                # concatenate input_i and future_candles[i]
                input_with_future = np.concatenate((input_i, future_candles[i]), axis=0)

                latent_vector = proj_output[i]
                predicted_label = find_label_of_closest_latent_vector(nn_map, labels_db, latent_vector, config)
                ATR = calculate_atr(input_i)
                if predicted_label is not None:  #and ATR > config['spread'] * 3
                    trade_made = True
                    if predicted_label in [0, 1]:  # only consider buy and sell
                        profit_made, profit_or_loss, trade_cancelled, stop_loss_hit, take_profit_hit = made_a_profit(predicted_label, future_candles[i], input_i, config, ATR)
                        if trade_cancelled:
                            n_cancelled_trades += 1
                        if stop_loss_hit:
                            n_trade_loss_with_hitting_stop_loss += 1
                        if take_profit_hit:
                            n_trade_profit_with_hitting_take_profit += 1
                        total_profit += profit_or_loss
                        if profit_made:
                            profit_only += profit_or_loss
                            n_profitable_trades += 1
                        else:
                            loss_only += profit_or_loss
                        if predicted_label == gt_label[i]:
                            number_of_correct_predictions += 1
                            if not use_prod_model:
                                plot_candle_prediction(input_i, input_with_future, predicted_label, total_predictions, 'correct', profit_made, ATR)
                        else:
                            if not use_prod_model:
                                plot_candle_prediction(input_i, input_with_future, predicted_label, total_predictions, 'incorrect', profit_made, ATR)
                        total_predictions += 1
                        # print(f"total predictions: {total_predictions}")
                        # print(f"profitable trades: {n_profitable_trades / total_predictions}")
                        # print(f"total profit: {total_profit}")


            # if a trade was made, hold for 6 hours
            if trade_made:
                trade = False
                hold_counter += 1
                if hold_counter == config['trading_frequency']:
                    trade_made = False
                    hold_counter = 0
                    trade = True

    percentage_profitable_trades = 0
    if total_predictions > 0:
        percentage_profitable_trades = n_profitable_trades / total_predictions

    total_profit = total_profit * config['profit_scale']
    print(f"Instrument {config['instrument']} - Total predictions: {total_predictions}, profitable trades: {percentage_profitable_trades}, total profit: {total_profit}")
    #print(f"Total trade opportunities: {total_trade_opportunities}")
    return percentage_profitable_trades, total_predictions, total_profit, n_cancelled_trades, \
        n_trade_loss_with_hitting_stop_loss, n_trade_profit_with_hitting_take_profit, profit_only, loss_only, n_profitable_trades


if __name__ == "__main__":
    run_test(config=None, use_prod_model=False)