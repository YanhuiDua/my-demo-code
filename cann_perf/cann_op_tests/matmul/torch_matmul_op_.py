import torch
import torch_npu
import torch.nn as nn
import time

# x = torch.rand(2, 3, 4).to("npu:0")
# # print(x.shape)
# y = torch.rand(3, 12, 64, 512).to("npu:0")
# # print(y.shape)

# z = torch.matmul(x, y)
# print(z) 

# for i in range(10):
#     output = torch.matmul(x, y)

# start_time = time.time()
# for i in range(100):
#     output = torch.matmul(x, y)
# end_time=time.time()
# print((end_time - start_time)/100)

# print("run_time: ", end_time -  start_time)

x = torch.rand(2, 3, 4,5).to("npu:0")
print(x)
size = x.size()
print(size)
x1 = x.view(-1, size[len(size) - 1])
print(x1.shape)


