from config.config import get_instruments
from trade import trade

if __name__ == "__main__":
    instruments = get_instruments()
    trade(instruments, default_account=False, skip_first_trade=False)

