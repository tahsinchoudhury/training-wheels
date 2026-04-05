from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

print(ds)
print(ds["train"][0])