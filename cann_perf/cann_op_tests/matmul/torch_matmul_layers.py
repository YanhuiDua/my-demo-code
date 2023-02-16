import torch
import torch_npu
import torch.nn as nn
import time
layer = nn.Linear(17,17).to("npu:0")
layer.weight = torch.nn.Parameter(torch.ones(17,17).to("npu:0"))
layer.bias = torch.nn.Parameter(torch.ones(17).to("npu:0"))

input = torch.ones(481,1921,17,17).to("npu:0")

# output = layer(input)

# print(output)

for i in range(10):
    output = layer(input)

start_time = time.time()
for i in range(100):
    output = layer(input)
end_time=time.time()
print((end_time - start_time)/100)