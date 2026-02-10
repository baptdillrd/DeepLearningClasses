import torch
import numpy as np

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print(f"Tensor.mul(tensor) \n {tensor.mul(tensor)} \n ")
print(f"tensor * tensor \n {tensor * tensor} \n ")

print(f"Tensor.matmul(tensor) \n {tensor.mul(tensor)} \n ")
print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n ")

print(tensor, "\n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t : {t}")
n = t.numpy()
print(f"n : {n}")

t.add_(1)
print(f"t : {t}")
print(f"n : {n}")

n = np.ones(5)
t = torch.from_numpy(n)


np.add(n, 1, out=n)
print(f"t : {t}")
print(f"n : {n}")


