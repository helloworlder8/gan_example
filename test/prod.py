import numpy as np
import torch

# 创建一个 NumPy 数组
array = np.array([1, 2, 3, 4, 5])

# 设置数组为不可写
array.flags.writeable = False

# 如果需要将不可写的数组传递给 PyTorch，可以先复制一份可写的数组
array_copy = np.copy(array)
tensor = torch.from_numpy(array_copy)

# 打印 tensor
print(tensor)
