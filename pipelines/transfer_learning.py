from torch import optim
from torch.nn import functional as F
from models.models import StonksNet
import torch

from utils.load_file import *

def fake_prices(size, num_samples):
    return [torch.rand(size) for _ in range(num_samples)]


def correlation_pipeline():
    index = 'nyse'
    component_names = index_component_names(index)
    prices_i = load_index_prices('^NYA')

    component_corr = {}

    for component in component_names:
        prices_c = component_prices(index, component)
        component_corr[component] = np.corrcoef(prices_i, prices_c)[0, 1]

    print(component_corr)
    outpath = f'stonks/relationships/correlation/{index}.csv'
    df = pd.DataFrame(list(component_corr.items()), columns=['stonk', 'correlation'])
    df.to_csv(outpath)


def train_stonksnet(model:StonksNet):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    samples = fake_prices(model.size, 1000)

    for epoch in range(20):
        for x_i in samples:
            y_hat = model(x_i)

            loss = F.mse_loss(input=y_hat, target=x_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate mse loss
        print(f'epoch {epoch} avg loss: {np.mean([F.mse_loss(X, model(X)).item() for X in samples])}')

    # run test
    test_samples = fake_prices(model.size, 100)
    losses = []

    for t in test_samples:
        y_hat = model(t)
        loss = F.mse_loss(y_hat, t)
        losses.append(loss.item())

    mse = np.mean(losses)
    print(f'average test loss: {mse}')

    return model


def mininet_pipeline():
    index = 'nyse'
    component_names = index_component_names(index)

    size = len(component_names)

    # model.load_state_dict(torch.load(f'weights/mininet_{index}.pt'))
    model = StonksNet(size=size)
    model = train_stonksnet(model)

    save_path = f'weights/mininet_{index}.pt'
    torch.save(model.state_dict(), save_path)

    component_mse = {}
    prices = []
    for symbol in component_names:
        series = component_prices(index, symbol)
        train, _ = train_test_splits(series)
        prices.append(train)

    prices = np.array(prices)
    # prices_trans = prices.T
    # prices_trans = torch.from_numpy(prices_trans)
    prices = torch.from_numpy(prices.T)

    # predictions_trans = model(prices_trans)
    # predictions = predictions_trans.numpy().T
    predictions = model(prices).detach().numpy().T
    prices = prices.T

    for i, component in enumerate(component_names):
        actual = prices[i]
        predicted = predictions[i]
        mse = F.mse_loss(actual, torch.tensor(predicted))
        component_mse[component] = mse.item()

    print(component_mse)

    outpath = f'stonks/relationships/stonksnet/{index}.csv'
    df = pd.DataFrame(list(component_mse.items()), columns=['stonk', 'MSE'])
    df.to_csv(outpath)

    return model

def gather_groups():
    group_sizes = {'djia':4, 'gspc':6, 'nasdaq':6, 'nyse':8}
    groups = {}


    for index in group_sizes:
        size = group_sizes[index]
        hs = int(size/2)
        group = {'correlation': {}, 'mse': {}}
        correlations = load_relationship_info('correlation', index)
        correlations = correlations.sort_values(by=['correlation'], ascending=False).to_numpy()
        correlations = list(map(lambda x: x[1], correlations))

        # high correlation, low correlation, half high + half low
        group['correlation']['high'] = correlations[0:size]
        group['correlation']['low'] = correlations[-size:]
        group['correlation']['highlow'] = correlations[-hs:] + correlations[0:hs]

        mse = load_relationship_info('stonksnet', index)
        mse = mse.sort_values(by=['MSE'], ascending=False).to_numpy()
        mse = list(map(lambda x: x[1], mse))

        group['mse']['high'] = mse[0:size]
        group['mse']['low'] = mse[-size:]
        group['mse']['highlow'] = mse[-hs:] + mse[0:hs]

        groups[index] = group

    return groups


if __name__ == '__main__':
    gather_groups()
