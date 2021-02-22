from pipelines.build_batches import batched
from models.models import NumQModel
from pipelines.train import train_numq


print("Preparing data...")
dataloader = batched('sx5e')

if __name__ == '__main__':
    lr = 1e-4
    model = NumQModel()
    threshold = 0.01
    gamma = 0.85
    strategy = 'HOLD'
    dataloader = batched(dataset='gspc', batch_size=64)

    train_numq(model, dataloader, threshold, gamma, lr, strategy)
