from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Agg') 这行代码的作用，就是明确告诉 Matplotlib：“不要尝试在屏幕上显示任何图表，直接把它们保存成文件
import matplotlib.pyplot as plt

def generate_sine_wave():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imp', '--implementation', type=str, default='custom', choices=['torch', 'custom'], help='Implementation types to use')
    parser.add_argument('--arch', type=str, default='gru', choices=['rnn', 'lstm', 'gru'], help='LSTM types to use')
    parser.add_argument('--epochs', type=int, default=15, help='epochs to run')
    #args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    print(args)
    
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = generate_sine_wave()
    # 0到第3行是测试集，3到最后是训练集，拿前999个数，预测后999个数。
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    
    # build the model
    if args.arch == "gru":
        from models import GRUNet as Net
    elif args.arch == "lstm":
        from models import LSTMNet as Net
    elif args.arch == "rnn":
        from models import RNNNet as Net
    else:
        raise ValueError("{} not implemented.".format(args.arch))

    seq = Net(args, input_size=1, hidden_dim=51)

    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    #begin to train
    
    for i in range(args.epochs):
        print('STEP: ', i)
        '''
        LBFGS 的 closure 可能被调用多次，是因为它内部需要用线搜索来找到最优步长。
        每尝试一个新步长，都要重新做一次前向传播和反向传播来计算损失和梯度。
        这是 LBFGS 比 Adam 更"精确"但也更"慢"的原因
        '''
        def closure():
            optimizer.zero_grad()
            out = seq(input) # 调用_call_函数，里面包含调用forward函数.但显式调用会绕过PyTorch 的钩子系统、分布式同步、梯度检查等核心功能
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('results/predict{}_{}_{}.png'.format(i, args.arch, args.imp))
        plt.close()