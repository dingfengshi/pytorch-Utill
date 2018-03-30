import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import DataUtill
from tensorboardX import SummaryWriter

# 学习参数
d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = (0.5, 0.999)
num_epochs = 30000
# 训练比例
d_steps = 1
g_steps = 1
model_path = ""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义模型参数

    def forward(self, x):
        # 定义网络结构
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义模型参数

    def forward(self, x):
        # 定义网络结构
        pass


def noise_sampler(size):
    return torch.FloatTensor(np.random.normal(size=size))


def train():
    G = Generator()
    D = Discriminator()
    G.cuda()
    D.cuda()
    # 定义损失
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

    # log目录
    writer = SummaryWriter(log_dir=model_path)

    # 取得数据集
    dataloader = DataUtill.get_dataloader("frame_file", "path", 64, False)

    for epoch in range(num_epochs):
        for step, batch_data in enumerate(dataloader):
            # 训练判别器
            for d_index in range(d_steps):
                D.zero_grad()

                # 训练真实图像
                d_real_data = batch_data["real_data"]
                d_real_decision = D(d_real_data)
                d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))
                writer.add_scalar("Discriminator_real_loss", d_real_error)
                d_real_error.backward()

                # 训练生成图像
                d_gen_input = Variable(noise_sampler(128))
                d_fake_data = G(d_gen_input).detach()  # 中断梯度
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))
                d_fake_error.backward()
                d_optimizer.step()

            for g_index in range(g_steps):
                # 训练generator
                G.zero_grad()

                gen_input = Variable(noise_sampler(128))
                g_fake_data = G(gen_input)
                dg_fake_decision = D(g_fake_data.t())
                g_error = criterion(dg_fake_decision,
                                    Variable(torch.ones(1)))
                g_error.backward()
                g_optimizer.step()

    writer.close()
