import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据集的转换
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 加载MNIST数据集
mnist_dataset = datasets.MNIST(root='./datasets/', train=True, download=True, transform=transform)

# 使用DataLoader加载数据集
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# 获取一批图像和标签
images, labels = next(iter(data_loader))

# 反归一化处理，将图像从[-1, 1]恢复到[0, 1]
images = images * 0.5 + 0.5

# 绘制几张图像进行展示
fig, axes = plt.subplots(1, 6, figsize=(15, 15))
for i in range(6):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {labels[i].item()}')
    axes[i].axis('off')

plt.show()
