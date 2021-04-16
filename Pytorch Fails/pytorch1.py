import torch
import numpy as np

data = [[1,2],[2,  4]]
x_data = torch.tensor(data)

# print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float,layout=torch.strided) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")
# print(x_rand.stride())
# print(x_rand.shape)


shape = (200,2,2,3,3) 
rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

print(rand_tensor.element_size())
# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
