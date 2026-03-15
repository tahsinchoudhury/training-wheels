import torch


values = torch.tensor([10, 20, 30])
repeated = torch.repeat_interleave(values, repeats=2)

print("torch.repeat_interleave() repeats each element in order.")
print("Input:", values)
print("repeats=2:", repeated)

# You can also repeat along a specific dimension.
matrix = torch.tensor([[1, 2], [3, 4]])
repeated_rows = torch.repeat_interleave(matrix, repeats=2, dim=0)

print("\nExample with dim=0:")
print(matrix)
print(repeated_rows)
