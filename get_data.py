from config.config import get_config
import requests
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone


def get_max_candles_per_request(config):
    if config['granularity'] == 'M5':
        return 1250
    elif config['granularity'] == 'H1':
        return 250
    else:
        raise ValueError("Granularity not supported")


def get_training_data(config=None, trading_mode=False):

    if config is None:
        config = get_config()

    instrument = config['instrument']

    max_candles_per_request = get_max_candles_per_request(config)
    instrument = config['instrument']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {config["api_key"]}',
    }

    # Fetch data for the past 6 years
    end_date = datetime.now(timezone('UTC'))
    start_date = (end_date - timedelta(days=int(config['train_data_years']*365))).replace(tzinfo=timezone('UTC'))

    # Initialize empty dataframe to store all candles
    df = pd.DataFrame()
    print(f"Fetching data from {start_date} to {end_date}")
    while start_date < end_date:
        params = (
            ('count', max_candles_per_request),
            ('granularity', config['granularity']),
            ('from', start_date.isoformat())
        )

        response = requests.get(config['api_url'] + f"instruments/{instrument}/candles", headers=headers, params=params)
        data = response.json()
        candles = data['candles']

        # Update the start_date to the time of the last fetched candle
        if len(candles) > 0:
            start_date = pd.to_datetime(candles[-1]['time'])
            # convert start_date to appropriate timezone so that it can be compared with end_date
            start_date = start_date.tz_convert(timezone('UTC'))
            # add 1 hour to the start_date to avoid fetching the same candle again
            start_date += timedelta(hours=1)
        else:
            break

        # save o,h,l,c to a dataframe
        df_temp = pd.DataFrame(candles)
        df_temp = df_temp[['time', 'mid']]
        df_temp['time'] = pd.to_datetime(df_temp['time'])
        df_temp['o'] = df_temp['mid'].apply(lambda x: x['o'])
        df_temp['h'] = df_temp['mid'].apply(lambda x: x['h'])
        df_temp['l'] = df_temp['mid'].apply(lambda x: x['l'])
        df_temp['c'] = df_temp['mid'].apply(lambda x: x['c'])
        df_temp = df_temp[['time', 'o', 'h', 'l', 'c']]

        # Append to the main dataframe
        df = pd.concat([df, df_temp], ignore_index=True)

    # Split data into three parts:
    # 1. most recent 1 month - test data
    # 2. 6 months ago to 1 month ago - validation data
    # 3. everything else - training data
    test_days = 30
    train_offset = test_days - 9
    if trading_mode:
        print("Using all data for training in trading mode")
        train_offset = 0
    else:
        print("Ignoring last few days of data for training")

    six_months_ago = end_date - timedelta(days=180)
    test_delta = end_date - timedelta(days=test_days)
    df_test = df[df['time'] > test_delta]
    df_val = df[(df['time'] > (end_date - timedelta(days=180))) & (df['time'] <= test_delta)]
    df_train = df[df['time'] <= (end_date - timedelta(days=train_offset))]

    # Save the data to csv files
    df_val.to_csv(config['val_data_dir'] + f'{instrument}.csv', header=True, index=False)
    df_train.to_csv(config['train_data_dir'] + f'{instrument}.csv', header=True, index=False)
    df_test.to_csv(config['test_data_dir'] + f'{instrument}.csv', header=True, index=False)


if __name__ == '__main__':
    get_training_data()