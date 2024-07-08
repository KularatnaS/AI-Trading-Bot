def get_instruments():
    # instruments = ['AUD_USD', 'EUR_USD', 'GBP_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'AUD_CAD', 'GBP_CAD', 'GBP_AUD',
    #                'EUR_AUD', 'EUR_GBP', 'NZD_CAD', 'GBP_CHF', 'AUD_NZD', 'EUR_CHF', 'EUR_CAD', 'AUD_CHF', 'GBP_NZD',
    #                'AUD_SGD', 'GBP_SGD', 'NZD_SGD', 'CAD_SGD', 'EUR_SGD',
    #                'AUD_JPY', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY']
    #                # 'XAU_USD']

    instruments = ['GBP_JPY']

    # assert that there are no duplicates
    assert len(instruments) == len(set(instruments))
    return instruments

def get_config(instrument, default_account):
    train_data_years = 19
    epochs = int(5 * (19 / train_data_years))

    if instrument.endswith('JPY'):   # ATR  is around 0.2 - 0.3
        spread = 0.03
        min_profit = 0.09
        order_decimals = 3
        trading_units = 10_000
        profit_scale = 1/100
    elif instrument == 'XAU_USD':
        spread     = 0.15
        min_profit = 5.0
        order_decimals = 3
        trading_units = 100
        profit_scale = 1/100
    else:
        spread     = 0.00025
        min_profit = 0.0010
        order_decimals = 5
        trading_units = 10_000
        profit_scale = 1

    if default_account:
        #print("Using default account")
        api_key = '3438d7ffc2815ce34947bd70e0a1892b-d0528f3236e416e0b54f2717626f009c'
        atr_exit_factor_profit = 2.0
        atr_exit_factor_loss = 1.0
    else:
        print("Using alternate account")
        api_key = '3c1ae5dcd59d424391529c5999bb9cb3-648f047725a161ac1ab6a51be878a4cb'
        atr_exit_factor_profit = 0.95
        atr_exit_factor_loss = 0.45

    return {
        # oanda parameters
        'api_url': 'https://api-fxpractice.oanda.com/v3/',
        'api_key': api_key,
        'instrument': instrument,
        'granularity': 'H1',
        'prediction_units': 48,
        'spread': spread,  # 2.5 pips
        'min_profit': min_profit,  # 10 pips
        'profit_scale': profit_scale,
        'order_decimals': order_decimals,
        'n_trade_decisions': 3,  # 0: buy, 1: sell, 2: hold
        'atr_exit_factor_profit': atr_exit_factor_profit,
        'atr_exit_factor_loss': atr_exit_factor_loss,
        'buy_sell_decision_min_prob': 0.6,  # 0.60
        'hold_min_prob': 0.7,  # 0.75
        'train_data_years': train_data_years,
        'trading_units': trading_units,
        # 'leverage': 30,
        'trading_frequency': 1,  # in hours
        'trade_cancel_check_frequency': 1/60,  # in hours, 1 minute

        # Data parameters
        'data_dir': 'data/',
        'train_data_dir': 'data/train/',
        'val_data_dir': 'data/val/',
        'test_data_dir': 'data/test/',
        'data_base_dir': 'databases/',
        # Training parameters
        'seq_len': 216,
        'd_model': 4,
        'd_ff': 1024,
        'latent_dims': 250,
        'N': 6,
        'dropout': 0.015,
        'batch_size': 512,
        'lr': 10**-4,
        'epochs': epochs,
        # model checkpoint
        'preload': None,  # None or epoch number
        'model_folder': 'weights',
        'model_basename': 'tmodel_' + instrument + '_',
        'experiment_name': 'runs/tmodel'
    }


def get_weights_file_path(model_folder, model_basename, epoch):
    # if epoch is a single digit, pad it with a zero
    if type(epoch) is int:
        if epoch is not None and epoch < 10:
            epoch = f'0{epoch}'
    return f'{model_folder}/{model_basename}{epoch}.pt'
