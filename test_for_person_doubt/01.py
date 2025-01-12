def split_line(char):
    line = ''
    for i in range(15):
        line += char
    print(line)


num_run = 100
curr_run = 'run%i' % num_run
print(curr_run)
split_line('-')

for i in range(0, 100, 10):
    print(i)
split_line('-')

import torch
import numpy as np

data = torch.arange(0, 20).view(10, 2)
print(data)

# obs[:, i]，你可以提取出所有行的第 i 列，即提取出所有智能体在当前时间步的观察结果
new_list = [torch.Tensor(np.vstack(data[:, i])) for i in range(data.shape[1])]
print(new_list)

split_line('-')
print(10 // 2)
split_line('-')
list1 = []
for i in range(5):
    list1.append(np.arange(32).reshape(1,32))

print(list1)
tensor_for_list1 = torch.tensor(list1)
print(tensor_for_list1.shape)
