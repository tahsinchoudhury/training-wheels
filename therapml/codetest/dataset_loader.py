from datasets import load_dataset
from torch.utils.data import DataLoader

DATASET_PATH = "roneneldan/TinyStories"

ds = load_dataset(DATASET_PATH)

# Convert to PyTorch format
ds["train"].set_format(type="torch")

train_loader = DataLoader(ds["train"], batch_size=32, shuffle=True)

for batch in train_loader:
    for i in range(len(batch["text"])):
        print(batch["text"][i])
        print("-" * 50)
    break