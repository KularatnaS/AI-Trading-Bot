import pickle
import numpy as np
import torch
import requests
import pandas as pd
import json
from pytz import timezone
from config.config import get_config, get_weights_file_path, get_instruments
from model.model import VAE

from sklearn.neighbors import NearestNeighbors
import time
from datetime import datetime, timedelta
from test import find_label_of_closest_latent_vector, get_nn_map_and_labels_db, get_model
from dataset.dataset import calculate_atr
import re

device = "cpu"

def cancel_pending_orders(config):
    headers = get_headers(config)
    account_id = get_account_id(config)
    response = requests.get(f'{config["api_url"]}accounts/{account_id}/pendingOrders', headers=headers)
    # print number of pending orders
    print(f"Number of pending orders: {len(response.json()['orders'])}")
    # for each pending order, print the time at which the order was placed as datetime object, original format is
    # '2024-03-18T13:18:23.373154410Z'
    for order in response.json()['orders']:
        creation_time = order['createTime']
        creation_time = re.sub(r'\.\d+Z', '', creation_time)
        creation_time = datetime.strptime(creation_time, '%Y-%m-%dT%H:%M:%S')
        pending_time = (datetime.now() - creation_time).seconds / 3600
        # if pending time is more than 48 hours, cancel the order
        if pending_time > config['prediction_units']:
            if order['state'] == 'PENDING':
                response = requests.put(f'{config["api_url"]}accounts/{account_id}/trades/{order["tradeID"]}/close',
                                        headers=headers)
                print(f"Closed trade: {order['tradeID']}")
        else:
            print(f"Order: {order['id']}, type: {order['type']}, time: {creation_time}, pending for: {np.round(pending_time, 2)} hours")

def get_headers(config):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {config["api_key"]}',
    }
    return headers
def get_model_input(config):
    instrument = config['instrument']
    headers = get_headers(config)
    params = (
        ('count', config['seq_len']),
        ('granularity', 'H1')
    )

    response = requests.get(config['api_url'] + f"instruments/{instrument}/candles", headers=headers, params=params)
    data = response.json()
    candles = data['candles']

    # save o,h,l,c to a dataframe
    df_temp = pd.DataFrame(candles)
    df_temp = df_temp[['time', 'mid']]
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp['o'] = df_temp['mid'].apply(lambda x: x['o'])
    df_temp['h'] = df_temp['mid'].apply(lambda x: x['h'])
    df_temp['l'] = df_temp['mid'].apply(lambda x: x['l'])
    df_temp['c'] = df_temp['mid'].apply(lambda x: x['c'])

    # print t,o,h,l,c of last candle#
    # print(f"Time of last candle: {df_temp.iloc[-1]['time']}")
    # print(f"Open: {df_temp.iloc[-1]['o']}")
    # print(f"High: {df_temp.iloc[-1]['h']}")
    # print(f"Low: {df_temp.iloc[-1]['l']}")
    # print(f"Close: {df_temp.iloc[-1]['c']}")
    # print time of last candle
    #print(f"Time of last candle: {df_temp.iloc[-1]['time']}")
    df_temp = df_temp[['o', 'h', 'l', 'c']]


    # calculate average true range
    atr = calculate_atr(df_temp.values.astype(np.float32))

    # return df_temp as numpy array, normalized and converted to float32
    out = df_temp.values.astype(np.float32)
    # reshape to (seq_len * d_model)
    out = out.reshape(-1)
    # normalize
    mean = out.mean()
    std = out.std()
    out = (out - mean) / std
    return torch.from_numpy(out).unsqueeze(0).to(device), atr


def list_pending_orders(config, headers):
    response = requests.get(config['api_url'] + 'accounts', headers=headers)
    account_info = response.json()
    account_id = account_info['accounts'][0]['id']
    response = requests.get(f'{config["api_url"]}accounts/{account_id}/pendingOrders', headers=headers)
    print(response.json())


def get_account_id(config):
    headers = get_headers(config)
    response = requests.get(config['api_url'] + 'accounts', headers=headers)
    account_info = response.json()
    account_id = account_info['accounts'][0]['id']
    return account_id
def place_order(config, atr, decision, units):
    headers = get_headers(config)
    account_id = get_account_id(config)
    response = requests.get(f'{config["api_url"]}accounts/{account_id}/pricing?instruments={config["instrument"]}',
                            headers=headers)
    current_price = response.json()['prices'][0]['bids'][0]['price']

    # place order
    data = None
    if decision == 0:
        data = {

            "order": {
                "units": str(units),
                "instrument": config["instrument"],
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {
                    "price": str(np.round(float(current_price) + config['atr_exit_factor_profit'] * atr,
                                          config['order_decimals']))
                },
                "stopLossOnFill": {
                    "price": str(np.round(float(current_price) - config['atr_exit_factor_loss'] * atr,
                                          config['order_decimals']))
                }
            }
        }
    elif decision == 1:
        data = {

            "order": {
                "units": str(-units),
                "instrument": config["instrument"],
                "timeInForce": "FOK",
                "type": "MARKET",

                "positionFill": "DEFAULT",
                "takeProfitOnFill": {
                    "price": str(np.round(float(current_price) - config['atr_exit_factor_profit'] * atr, config['order_decimals']))
                },
                "stopLossOnFill": {
                    "price": str(np.round(float(current_price) + config['atr_exit_factor_loss'] * atr, config['order_decimals']))
                }
            }
        }

    if data is not None:
        response = requests.post(f'{config["api_url"]}accounts/{account_id}/orders', headers=headers,
                                 data=json.dumps(data))
        # print(response.json())


def pause_trading(config):
    while True:
        if check_time_is_between_two_minute_values(0, 1):
            break
        else:
            # print in same line the remaining time to next trade in minutes
            print(f"\rWaiting to break the loop", end='')

def check_time_is_between_two_minute_values(start_time, end_time):
    """
    Given start_time and end_time in minutes as integers ranging from 0-60,check if the current time is between
    start_time and end_time and return True if it is, else return False
    """

    current_time = datetime.now().time()
    current_time_minutes = current_time.minute
    if start_time <= current_time_minutes <= end_time:
        return True
    return False

def wait_for_next_trade_interval(config):
    time_spent_in_seconds = 0
    time_delta = 5
    while True:
        time.sleep(time_delta)
        if check_time_is_between_two_minute_values(58, 59):
            print("Trading interval reached")
            break
        else:
            # print in same line the remaining time to next trade in minutes
            print(f"\rNext trade in {58 - datetime.now().time().minute} minutes", end='')
        time_spent_in_seconds += time_delta
        if time_spent_in_seconds > 60:
            try:
                cancel_pending_orders(config)
                time_spent_in_seconds = 0
            except Exception as e:
                print(e)

def trade(instruments, default_account, skip_first_trade):
    config = get_config('EUR_USD', default_account)  # get config for any instrument
    n_orders = 0
    decisions_made = []
    with torch.no_grad():
        while True:
            if skip_first_trade:
                print("Skipping first trade")
                skip_first_trade = False
                pause_trading(config)
            else:
                wait_for_next_trade_interval(config)
                for instrument in instruments:
                    config = get_config(instrument, default_account)
                    print(f"Trading for {instrument} with units: {config['trading_units']}")
                    model = get_model(config, device, use_prod_model=True)
                    model.eval()
                    nn_map, labels_db = get_nn_map_and_labels_db(config)
                    model_input, atr = get_model_input(config)
                    latent_vector = model.encoder(model_input)
                    latent_vector = latent_vector.cpu().numpy()[0]
                    decision = find_label_of_closest_latent_vector(nn_map, labels_db, latent_vector, config)
                    decisions_made.append(f'{instrument}: {decision}')
                    if decision is not None:
                        n_orders += 1
                    # print(f"Latest decision: {decision}, Total orders placed: {n_orders}")
                    try:
                        place_order(config, atr, decision, units=config['trading_units'])
                    except Exception as e:
                        print("Failed to place order")
                        print(e)
                    # sleep for 3 hour and print remaining time to next trade in minutes in a single line, where the previous line is overwritten
                print(f"Decisions made: {decisions_made}")
                pause_trading(config)



if __name__ == "__main__":
    instruments = get_instruments()
    trade(instruments)



