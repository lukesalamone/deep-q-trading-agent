from .build_batches import load_prices



class FinanceEnvironment:
    def __init__(self, prices):
        self.prices = prices
        self.time_step = 0

    def step(self):
        self.time_step += 1

def make_env(symbol):
    prices = load_prices(symbol)
    return FinanceEnvironment(prices)