import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 示例用户数据
a = np.array([[1, 2], [2, 3], [2, 3], [1, 5], [2, 5]])
b = np.array([[3, 5], [2, 4], [5, 4], [1, 2], [3, 4]])
c = np.array([[1, 4], [2, 3], [2, 5], [1, 5], [3, 5]])
d = np.array([[3, 5], [2, 4], [5, 4], [4, 2], [2, 4]])
e = np.array([[2, 4], [1, 3], [2, 2], [1, 4], [2, 5]])
f = np.array([[1, 4], [2, 2], [3, 4], [3, 2], [1, 3]])
user_data = {'user1': {'video1': a, 'video2': b}, 'user2': {'video1': c, 'video2': d}, 'user3': {'video1': e, 'video2': f}}

# 准备训练数据
train_data = []
for user in user_data:
    for video in user_data[user]:
        train_data.append(user_data[user][video].flatten())

train_data = np.array(train_data)
train_data = torch.tensor(train_data, dtype=torch.float32)

# 生成器定义
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# 判别器定义
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 训练参数
input_size = train_data.shape[1]  # 输入序列的长度
hidden_size = 128
batch_size = 5
num_epochs = 10000
lr = 0.0002

# 初始化生成器和判别器
G = Generator(input_size, hidden_size, input_size)
D = Discriminator(input_size, hidden_size)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# 训练GAN
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        real_data = train_data[i:i + batch_size]
        batch_size_current = real_data.shape[0]

        # 训练判别器
        optimizer_D.zero_grad()
        
        # 真实数据损失
        real_labels = torch.ones(batch_size_current, 1)
        outputs = D(real_data)
        d_loss_real = criterion(outputs, real_labels)
        
        # 噪声数据损失
        noise = torch.randn(batch_size_current, input_size)
        fake_data = G(noise)
        fake_labels = torch.zeros(batch_size_current, 1)
        outputs = D(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        # 总判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        
        # 生成器损失
        outputs = D(fake_data)
        g_loss = criterion(outputs, real_labels)
        
        g_loss.backward()
        optimizer_G.step()

    # 打印损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# 生成新序列
fixed_noise = torch.randn(1, input_size)
new_sequence = G(fixed_noise).detach().numpy().reshape(-1, 2)
print("Generated sequence:\n", new_sequence)