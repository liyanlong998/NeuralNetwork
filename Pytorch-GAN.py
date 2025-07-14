import numpy as np
import scipy.special
import matplotlib.pyplot as plt

#定义了一个神经网络类，其中有4个参数
class neuralNetwork:
    def __init__ (self,input_nodes,hidden_nodes,output_nodes,learning_rate):

        #将传入的参数，保存为类的属性，以便后续使用
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        #采用scipy库来定义一个激活函数
        self.activation_function = lambda x :scipy.special.expit(x)
        
        #创建权重矩阵，使用正态概率分布采样权重，平均值为零，标准方差为节点传入链接数目的开方
        #self.wih的维度是（隐藏层节点数*输入层节点数）
        #self.who的维度是（输出层节点数*隐藏层节点数）
        self.wih = np.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))

    def train(self, input_list,targets_list):
        inputs = np.array(input_list, ndmin = 2).T
        targets = np.array(targets_list,ndmin = 2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #最终误差
        output_errors = targets - final_outputs
        #隐藏层的误差
        hidden_errors = np.dot(self.who.T,output_errors)

        #输出层到隐藏层的权重更新
        self.who += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        #隐层层到输入的权重更新
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        pass

    def query(self,inputs_list):

        #定义一个输入矩阵，并使其转置
        inputs = np.array(inputs_list, ndmin = 2).T

        #输出层矩阵与相对应的权重矩阵相乘
        hidden_inputs = np.dot(self.wih,inputs)

        #隐藏层的输入值再带入激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        #隐藏层输出值与相对应的权重矩阵想乘
        final_inputs = np.dot(self.who,hidden_outputs)

        #输出值再带入激活函数
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass

input_nodes = 784
hidden_nodes = 100
outputs_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes,hidden_nodes,outputs_nodes,learning_rate)

data_file = open("F:\pyhton3\mnist_train_100.csv",'r')
data_list = data_file.readlines()
data_file.close()


for record in data_list:
    all_values = record.split(',')
    #将颜色值0~255的范围，调整到0.01~1.00
    inputs = ((np.asarray(all_values[1:],dtype=np.float32))/255.0*0.99)+0.01
    #节点数据最大，意味着节点被激活了
    targets = np.zeros(outputs_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
print("训练完成")



test_file = open("F:\pyhton3\mnist_test_10.csv",'r')
test_list = test_file.readline()
test_file.close
test_values = test_list.split(',')

image_array =np.asarray(test_values[1:],dtype=np.float32).reshape((28,28))
plt.imshow( image_array,cmap='Greys',interpolation='None')
plt.show()
print(test_values[0])
print(n.query(((np.asarray(test_values[1:],dtype=np.float32))/255.0*0.99)+0.01))
print("测试完成")


