from get_data import get_training_data
from config.config import get_config
from train import train_model
from create_db import create_db
from test import run_test
from config.config import get_instruments
from trade import trade

##############
def train(instruments, trading_mode, default_account):
    final_profitable_trades_percentage = 0
    final_trade_count = 0
    total_profit = 0
    n_cancelled_trades = 0
    n_stop_loss_trades = 0
    n_take_profit_trades = 0
    profit = 0
    loss = 0
    n_loss_making_trades = 0
    total_n_profitable_trades = 0
    for instrument in instruments:
        config = get_config(instrument, default_account)
        print(f"Training model for {instrument} with config: {config}")
        # get_training_data(config=config, trading_mode=trading_mode)
        # train_model(config=config, save_only_prod_epoch=True)
        create_db(config=config, use_prod_model=True)
        profitable_trade_percentage, trade_count, profit, cancelled_trades, stop_loss_trades, take_profit_trades, \
            profit_only, loss_only, n_profitable_trades = run_test(config=config, use_prod_model=True)
        final_profitable_trades_percentage += profitable_trade_percentage * trade_count
        final_trade_count += trade_count
        total_profit += profit
        n_cancelled_trades += cancelled_trades
        n_stop_loss_trades += stop_loss_trades
        n_take_profit_trades += take_profit_trades

        n_loss_making_trades += (trade_count - n_profitable_trades)
        total_n_profitable_trades += n_profitable_trades
        profit += profit_only
        loss += loss_only

        if total_n_profitable_trades > 0:
            average_profit = profit / total_n_profitable_trades
        if n_loss_making_trades > 0:
            average_loss = loss / n_loss_making_trades

    print(f"Final profitable trades percentage: {final_profitable_trades_percentage / final_trade_count}")
    print(f"Final trade count: {final_trade_count}")
    print(f"n cancelled trades: {n_cancelled_trades}")
    print(f"n stop loss trades: {n_stop_loss_trades}")
    print(f"n take profit trades: {n_take_profit_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit: {average_profit}")
    print(f"Average loss: {average_loss}")


if __name__ == "__main__":
    trading_mode = False  # forces all training data to be used if set to True, else ignore last few days of data
    instruments = get_instruments()
    train(instruments, trading_mode, default_account=True)
    #trade(instruments, default_account=True, skip_first_trade=False)

