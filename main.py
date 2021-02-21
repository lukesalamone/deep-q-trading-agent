from pipelines.build_batches import batched
from models.models import NumQModel

print("Preparing data...")
dataloader = batched('sx5e')

