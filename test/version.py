import torch

# 检查 PyTorch 版本
print(torch.__version__)

# # 使用最新版本的 PyTorch，不需要显式使用 Variable
# # real_img = Variable(imgs).cuda()  # 旧的写法
# real_img = imgs.cuda()  # 新的写法

# # 定义真实和假的标签
# real_label = torch.ones(imgs.size(0), 1).cuda()
# fake_label = torch.zeros(imgs.size(0), 1).cuda()
