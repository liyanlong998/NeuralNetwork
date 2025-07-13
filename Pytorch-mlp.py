import torch
import torch.nn as nn
import pandas
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

#nn.Module 是 PyTorch 中所有神经网络模块的基类（父类），你的自定义模型（如 classifier)必须继承它，
#PyTorch 才能正确管理模型的参数(如权重、偏置)、实现自动求导(Autograd)并提供模型保存/加载等功能。
class classifier(nn.Module):
    def __init__(self):
        #初始化pyTorch父类
        super().__init__()

        #创建一个序列模型容器，会按照顺序执行添加的层
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,10),
            nn.Sigmoid()
            )
        #创建损失函数
        self.loss_function = nn.BCELoss()
        #创建优化器，使用简单的梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(),lr = 0.01)
        
        #定义计数器
        self.counter = 0
        # 存储损失值的列表，用于后续分析和可视化
        self.progress = []
   
    #向前传播
    def forward(self, inputs):
        return self.model(inputs)
    

    #定义训练函数，附带记录功能，记录的是损失值
    def train(self, inputs, targets):
        #计算网路输出值
        outputs = self.forward(inputs)
        #计算损失值
        loss = self.loss_function(outputs,targets)
        
    
        # 每处理1个样本，计数器增加1
        self.counter += 1
        # 每10个样本，将损失值添加到进度列表
        # 使用 .item()将PyTorch张量转换为Python浮点数。换句话说是从单元素张量提取标量值
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        # 每10000个样本，打印当前计数器值   
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        #每个样本应该独立计算梯度，所以需要先清零。让模型基于当前数据学习，不受历史数据影响
        self.optimiser.zero_grad()
        #计算损失函数梯度
        loss.backward()
        #根据梯度更新参数
        self.optimiser.step()

    #定义绘图函数，将损失值绘制出来
    def plot_progress(self):
    # 将损失列表转换为 Pandas Data Frame , 列表名为"loss"  
        df = pandas.DataFrame(self.progress, columns=['loss'])
        # ylim=(0, 1.0),Y轴显示范围固定为[0,1]
        # figsize=(16,8),图像尺寸为16x8英寸
        # alpha=0.1，线条透明度10%（突出趋势而非噪声）
        # marker='.' ，每个数据点，用点标记
        # grid = true , 显示网格线
        # yticks=(0, 0.25, 0.5)，Y轴显示刻度的位置
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', 
        grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

#创建一个 Mnist Dataset 类，继承 Dataset 类，主要用来读取和处理数据
class MnistDataset(Dataset):

    # 读取数据，csv没有列标题行
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    #返回数据集中的项目总数
    def __len__(self):
        return len(self.data_df)
        
    #返回数据集中的第n项，获取单个样本
    def __getitem__(self, index):
        #.iloc[index,0]获取第index行，第0个值
        label = self.data_df.iloc[index,0]
        #创建长度为10的全零张量（对应0-9十个数字类别）
        target = torch.zeros((10))
        target[label] = 1.0

        # 图像数据, 取值范围是0~255，标准化为0~1
        # [index,1:]获取第index行从第1列开始的所有像素
        #.values 转换为Numpy数组
        # torch.FloatTensor（）转换为浮点型张量
        # /255 归一化
        image_values = torch.FloatTensor(self.data_df.iloc [index,1:].values) / 255.0
        # 返回标签、图像数据张量以及目标张量
        return label, image_values, target
    
    #可绘制图像
    def plot_image(self, index):
        arr = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(arr, interpolation='none', cmap='Blues')
        plt.show()
        pass

mnist_dataset = MnistDataset("E:\\Vscodeprojects\\mnist_train.csv")

C = classifier()
epochs = 3
for i in range(epochs):
    print('training epoch', i+1, "of", epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
        

C.plot_progress()

# 加载MNIST测试数据
mnist_test_dataset = MnistDataset("E:\\Vscodeprojects\\mnist_test.csv")
# 挑选一幅图像
record = 19
# 绘制图像和标签
mnist_test_dataset.plot_image(record)

image_data = mnist_test_dataset[record][1]
output = C.forward(image_data)
pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
plt.show()



score = 0
items = 0

for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(image_data_tensor).detach().numpy()
    if (answer.argmax() == label):
        score += 1
        pass
    items += 1

    pass

print(score, items, score/items)