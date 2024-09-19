# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
writer = SummaryWriter()

# %% 配置参数
NUM_AVERAGE       = 1
NUM_TRAIN         = 512
NUM_INTERVAL      = 50
DIMENSION         = 100
LEARNING_RATE     = 0.001
NUM_ITERATION     = 20000
FREQUENCY         = 100
NUM_TEST          = 100
NUM_TEST_INTERVAL = 100
FLAG              = 1

record_path = "FBSTD/01BS/Data/"
record_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
os.makedirs(record_path, exist_ok=True)

# %% 配置方程
class BS(object):
    def __init__(self, D):
        self.Xi               = torch.FloatTensor(np.array([1.0, 0.5] * int(D / 2))[None, :]).to(device)
        self.Xi.requires_grad = True
        self.T                = 1.0
        self.D                = D
        self.training_loss    = []
        self.iteration        = []

    def phi(self, t, X, Y, Z):
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True)).to(device)
    
    def terminal_cond(self, X):
        return torch.sum(X ** 2, 1, keepdim=True).to(device)
    
    def terminal_grad(self, X):
        g = self.terminal_cond(X)
        Dg = torch.autograd.grad(outputs      = [g], 
                                 inputs       = [X], 
                                 grad_outputs = torch.ones_like(g),
                                 allow_unused = True,
                                 retain_graph = True,
                                 create_graph = True)[0]
        return Dg

    def mu(self, t, X):
        M = X.shape[0]
        D = X.shape[1]
        return torch.zeros([M, D]).to(device)
    
    def sigma(self, t, X):
        return 0.4 * torch.diag_embed(X).to(device)

    def u_exact(self, t, X):
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max ** 2) * (self.T - t)) * np.sum(X ** 2, 1, keepdims=True)
# %% 构造网络
class Resnet(nn.Module):
    def __init__(self, dim):
        super(Resnet, self).__init__()
        self.layer1 = nn.Linear(dim+1, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        self.epsilon = 0.01

    def forward(self, x):
        
        out = self.layer1(x)
        out = torch.sin(out)
        # out = torch.cos(out)
        # out = out * torch.sigmoid(out)
        # out = F.relu(out)
        # out = F.tanh(out)

        for i in range(2):
            shortcut = out
            out = self.layer2(out)
            out = torch.sin(out)
            # out = torch.cos(out)
            # out = out * torch.sigmoid(out)
            # out = F.relu(out)
            # out = F.tanh(out)
            out = out + shortcut

        out = torch.sin(out)
        # out = torch.cos(out)
        # out = out * torch.sigmoid(out)
        # out = F.relu(out)
        # out = F.tanh(out)
        out = self.layer3(out)
        return out

class ResnetZ(nn.Module):
    def __init__(self, dim):
        super(ResnetZ, self).__init__()
        self.layer1 = nn.Linear(dim+1, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, dim)
        self.epsilon = 0.01

    def forward(self, x):

        out = self.layer1(x)
        out = torch.sin(out)
        # out = torch.cos(out)
        # out = out * torch.sigmoid(out)
        # out = F.relu(out)
        # out = F.tanh(out)

        for i in range(2):
            shortcut = out
            out = self.layer2(out)
            out = torch.sin(out)
            # out = torch.cos(out)
            # out = out * torch.sigmoid(out)
            # out = F.relu(out)
            # out = F.tanh(out)
            out = out + shortcut

        out = torch.sin(out)
        # out = torch.cos(out)
        # out = out * torch.sigmoid(out)
        # out = F.relu(out)
        # out = F.tanh(out)
        out = self.layer3(out)
        return out

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def net_u(t, X, model, flag):
    input = torch.cat((t, X), 1)
    u = model(input)
    if flag:
        Du = torch.autograd.grad(
            outputs      = [u], 
            inputs       = [X],
            grad_outputs = torch.ones_like(u),
            allow_unused = True,
            retain_graph = True,
            create_graph = True)[0]
        return u, Du
    else:
        return u

# %% 额外函数

def brown_motion(Eq, num_walkers, num_interval):
    M = num_walkers
    N = num_interval
    D = Eq.D
    Delta_t = Eq.T / N 
    Dt = np.zeros((M, N + 1, 1))
    DW = np.zeros((M, N + 1, D))
    
    Dt[:, 1:, :] = Delta_t
    DW[:, 1:, :] = np.sqrt(Delta_t) * np.random.normal(size=(M, N, D))

    t = np.cumsum(Dt, axis=1)
    W = np.cumsum(DW, axis=1)
    t = torch.FloatTensor(t).to(device)
    W = torch.FloatTensor(W).to(device)
    return t, W

def get_trajectory(Eq, t, W):
    M = W.shape[0]
    N = W.shape[1] - 1
    D = W.shape[2]

    t0 = t[:, 0, :]
    W0 = W[:, 0, :]
    
    # 随机起点
    # X0 = np.random.random(size=(M,D))
    # X0 = torch.FloatTensor(X0).to(device)
    # X0.requires_grad = True

    # 固定起点
    X0 = Eq.Xi
    X0 = X0.repeat(M, 1).view(M, D)

    X_list = X0.unsqueeze(1)

    for n in range(N):
        t1 = t[:, n + 1, :]
        W1 = W[:, n + 1, :]
        X1 = X0 + Eq.mu(t0, X0) * (t1 - t0) + torch.squeeze(torch.matmul(Eq.sigma(t0, X0), (W1 - W0).unsqueeze(-1)), dim=-1)

        t0 = t1
        W0 = W1
        X0 = X1
        X_list = torch.cat((X_list, X0.unsqueeze(1)), dim=1)
    return X_list

def analysis(file_path):
    train_history = pd.read_csv(file_path)
    time_step = train_history.iloc[0:, 0]
    L2error = train_history.iloc[0:, 4]

    print(f'L2error Mean:{np.mean(L2error)}, L2error Min:{np.min(L2error)}')

    return time_step, L2error

def save_model(Y_net, Z_net=None, subfix=''):
    torch.save(Y_net.state_dict(), f'{record_path}{Eq.D}D-{record_time}-Y_net{subfix}')
    print('Model Y Saved!')
    if not FLAG and Z_net is not None:
        torch.save(Z_net.state_dict(), f'{record_path}{Eq.D}D-{record_time}-Z_net{subfix}')
        print('Model Z Saved!')

# %% 开始训练
Eq = BS(DIMENSION)

MIN_L2 = 1.0
STEP = 0

for total in range(0, NUM_AVERAGE):
    torch.cuda.empty_cache()
    Y_net = Resnet(DIMENSION).to(device)
    Y_net.apply(weights_init)
    optY = optim.Adam(Y_net.parameters(), lr=LEARNING_RATE)

    if not FLAG:
        print('Z初始化')
        Z_net = ResnetZ(DIMENSION).to(device)
        Z_net.apply(weights_init)
        optZ = optim.Adam(Z_net.parameters(), lr=LEARNING_RATE)

    M = NUM_TRAIN
    N = NUM_INTERVAL
    D = DIMENSION
    Loss = 0
    training_history = []
    start_time = time.time()

    # for step in tqdm(range(NUM_ITERATION + 1)):
    for step in range(NUM_ITERATION + 1):
        if step and (step / 5000) % 2 == 0:
            for p in optY.param_groups:
                p['lr'] *= 0.5
                print(p['lr'])
            if not FLAG:
                for p in optZ.param_groups:
                    p['lr'] *= 0.5
        if step and (step / 5000) % 2 == 1:
            for p in optY.param_groups:
                p['lr'] /= 5
                print(p['lr'])
            if not FLAG:
                for p in optZ.param_groups:
                    p['lr'] /= 5

        # if step and step % 200 == 0:
            # for p in optY.param_groups:
            #     p['lr'] *= 0.95
            #     print(p['lr'])
            # if not FLAG:
            #     for p in optZ.param_groups:
            #         p['lr'] *= 0.95

        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            Y0_pred  = net_u(torch.zeros(size=(1,1)).to(device), Eq.Xi, Y_net, 0)
            Y0_pred  = Y0_pred.cpu().detach().numpy()[0][0]
            Y0_exact = Eq.u_exact(0, Eq.Xi.cpu().detach().numpy())
            Y0_L2err = np.sqrt(np.mean((Y0_exact-Y0_pred)**2)/np.mean(Y0_exact**2))
        
            # 固定起始误差
            np.random.seed(512)
            t, Wt    = brown_motion(Eq, NUM_TEST, NUM_TEST_INTERVAL)
            Xt       = get_trajectory(Eq, t, Wt)
            # 网络预测
            t        = t.reshape(-1, 1)
            Xt       = Xt.reshape(-1, Eq.D)
            Yt_pred  = net_u(t, Xt, Y_net, 0)
            Yt_pred  = Yt_pred.cpu().detach().numpy()
            # 计算真解
            t        = t.cpu().detach().numpy()
            Xt       = Xt.cpu().detach().numpy()
            Yt_exact = Eq.u_exact(t, Xt)
            Yt_L2err = np.sqrt(np.mean((Yt_exact-Yt_pred)**2)/np.mean(Yt_exact**2)) # 相对误差均值
            # 相对误差
            t_seq        = t.reshape(NUM_TEST, -1)
            Yt_pred_seq  = Yt_pred.reshape(NUM_TEST, -1)
            Yt_exact_seq = Yt_exact.reshape(NUM_TEST, -1)
            Yt_L2err_seq = np.sqrt((Yt_pred - Yt_exact)**2/(Yt_exact)**2)

            if Yt_L2err < 0.01 and Yt_L2err < MIN_L2:
                save_model(Y_net, subfix=f"-step_{step}-err_{Yt_L2err}")
                MIN_L2 = Yt_L2err
                STEP = step
                print(f'Minimum L2 Error: {MIN_L2} at Step: {STEP}')

            tqdm.write(
                f'\nStep: {step:<5}, '\
                f'Loss: {Loss:>10.5f}, '\
                f'Y0: {Y0_pred:.5f}, '\
                f'Y0_L2: {Y0_L2err:.5f},'\
                f'Yt_L2: {Yt_L2err:.5f}, '\
                f'Time: {elapsed_time:.2f}')

            writer.add_scalar('Loss', Loss, step)
            writer.add_scalar('Y0', Y0_pred, step)
            writer.add_scalar('Y0_L2', Y0_L2err, step)
            writer.add_scalar('Yt_L2', Yt_L2err, step)
            if type(Loss) == torch.Tensor:
                Loss_scalar = Loss.cpu().detach().numpy()
            else:
                Loss_scalar = Loss 
            training_history.append([step, Loss_scalar, Y0_pred, Y0_L2err, Yt_L2err, elapsed_time])

        if step % 20000 == 0:
            L2err = np.concatenate((t_seq.reshape(-1,1), Yt_L2err_seq.reshape(-1,1)),axis=1)
            L2err = DataFrame(L2err, columns=['Time', 'Relative Error'])

            plt.figure()
            is_exact = hasattr(Eq, 'u_exact')
            if is_exact:
                plt.plot(t_seq[:5, :].T, Yt_exact_seq[:5, :].T, 'r-')
            plt.plot(t_seq[:5, :].T, Yt_pred_seq[:5, :].T, 'bo--', markerfacecolor = 'w')
            if is_exact:
                plt.plot(t_seq[:5, :].T, Yt_exact_seq[:5, :].T, 'r.')

            plt.figure()
            sns.relplot(x='Time', y='Relative Error', kind='line', data=L2err)
            plt.show()

        Loss = 0
        if step == 0 or Eq.T - T0[0] < 1e-6:
            T0 = np.zeros((M, 1))
            X0 = Eq.Xi
            X0 = X0.repeat(M, 1)
            T0 = torch.FloatTensor(T0).to(device)

            # N = np.random.randint(1,100)
            Delta_t = Eq.T / N
            Dt = Delta_t * torch.ones((M, 1)).to(device) 
            if FLAG:
                Y0_fp, Z0 = net_u(T0, X0, Y_net, FLAG)
            else:
                Y0_fp = net_u(T0, X0, Y_net, FLAG)
                Z0 = net_u(T0, X0, Z_net, FLAG)
        
        if FLAG:
            Y0, Z0 = net_u(T0, X0, Y_net, FLAG)
        else:
            Y0 = net_u(T0, X0, Y_net, FLAG)
            Z0 = net_u(T0, X0, Z_net, FLAG)
        
        DW0 = np.sqrt(Delta_t) * np.random.normal(size=(M, D))
        DW0 = torch.FloatTensor(DW0).to(device)
        T1 = T0 + Dt
        X1 = X0 + Eq.mu(T0, X0) * Dt + torch.squeeze(torch.matmul(Eq.sigma(T0, X0), DW0.unsqueeze(-1)), dim=-1)

        Y1_fp = Y0 + Eq.phi(T0, X0, Y0, Z0) * Delta_t + torch.sum(Z0 * torch.squeeze(torch.matmul(Eq.sigma(T0, X0), DW0.unsqueeze(-1))), dim=1, keepdim=True)
        # Y1_fp = Y0_fp + Eq.phi(T0, X0, Y0_fp, Z0) * Delta_t + torch.sum(Z0 * torch.squeeze(torch.matmul(Eq.sigma(T0, X0), DW0.unsqueeze(-1))), dim=1, keepdim=True)

        # --- 倒向格式 ---
        # Y1_fp = Y0 + Eq.phi(T0, X0, Y0, Z0) * Delta_t
        # Z1_fp = torch.sum(Z0 * torch.squeeze(torch.matmul(, DW0.unsqueeze(-1))), dim=1, keepdim=True)/(Eq.sigma(T0, X0) * Delta_t)

        if FLAG:
            Y1, Z1 = net_u(T1, X1, Y_net, FLAG)
        else:
            Y1 = net_u(T1, X1, Y_net, FLAG)
            Z1 = net_u(T1, X1, Z_net, FLAG)
            
        # Y1_fp = Y0 + 0.5 * (Eq.phi(T0, X0, Y0, Z0) +  Eq.phi(T1, X1, Y1, Z1)) * Delta_t + torch.sum(Z0 * torch.squeeze(torch.matmul(Eq.sigma(T0, X0), DW0.unsqueeze(-1))), dim=1, keepdim=True)

        Loss += torch.sum(torch.pow(Y1-Y1_fp, 2))

        if step >= N-1:
            if Eq.T - T1[0] < 1e-6:
                XT = X1.detach().clone()
                XT.requires_grad = True
                TT = T1.detach().clone()
            if FLAG:
                YT, ZT = net_u(TT, XT, Y_net, FLAG)
            else:
                YT = net_u(TT, XT, Y_net, FLAG)
                ZT = net_u(TT, XT, Z_net, FLAG)
            Loss += 1/N * torch.sum(torch.pow(YT - Eq.terminal_cond(XT), 2))
            Loss += 1/N * torch.sum(torch.pow(ZT - Eq.terminal_grad(XT), 2)) 

        # if Eq.T - T1[0] < 1e-6:
        #     XT = X1.detach().clone()
        #     XT.requires_grad = True
        #     TT = T1.detach().clone()
        # if FLAG:
        #     YT, ZT = net_u(TT, XT, Y_net, FLAG)
        # else:
        #     YT = net_u(TT, XT, Y_net, FLAG)
        #     ZT = net_u(TT, XT, Z_net, FLAG)
        # Loss += torch.sum(torch.pow(YT - Eq.terminal_cond(XT), 2))
        # Loss += torch.sum(torch.pow(ZT - Eq.terminal_grad(XT), 2)) 

        optY.zero_grad()
        if not FLAG:
            optZ.zero_grad()
        Loss.backward(retain_graph=True)
        optY.step()
        if not FLAG:
            optZ.step()
        T0 = T1
        X0 = X1
        # Y0_fp = Y1_fp.detach().clone()

    training_history =np.array(training_history)
    if FLAG:
        num_net = 1
    else:
        num_net = 2
    file_name = f"{Eq.D}D-{num_net}Net-{NUM_ITERATION}-{NUM_INTERVAL}N-{LEARNING_RATE}lr-{record_time}.csv"
    np.savetxt(os.path.join(record_path, file_name), training_history, delimiter=",", header="step, loss, Y0, Y0_L2error, Yt_L2error, elapsed_time", comments='')
    print(f'Training History Saved as {file_name}!')
