from datasets import load_dataset

dataset = load_dataset("cais/mmlu", "all")

print(dataset)
print(dataset["auxiliary_train"][0])
print(dataset["auxiliary_train"][1])
print(dataset["auxiliary_train"][2]['answer'])
print(dataset["auxiliary_train"][2]['answer'])