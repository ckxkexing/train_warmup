import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import GoogleNet
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#对读取数据做个处理，打个包
#Normalize是数据归一化

bs = 128
epochs = 50
print("batch size = ", bs)
print("epochs = ", epochs)
trainset = torchvision.datasets.CIFAR10(root='../', train=True,
                                        download=True, transform=transform)
#封装了一个内置数据集调用的函数，初始化数据时会自动把数据拉下来拉到当前路径下
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,#表示PILImage的大小
                                          shuffle=True, num_workers=2) 
#加载测试集
testset = torchvision.datasets.CIFAR10(root='../', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs ,
                                         shuffle=False, num_workers=2)
#指定类别
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = GoogleNet()
# net.load_state_dict(torch.load('googlenet.pth'))
device = 'cuda:1' 
net.to(device)


import torch.optim as optim#做优化器
criterion = nn.CrossEntropyLoss()#分类问题用交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)#定义优化器（随机梯度下降）
# optimizer = optim.Adam(net.parameters(), lr=0.001)#定义优化器（随机梯度下降）

def acc(net):
    correct = 0
    total = 0
    with torch.no_grad():#不需要梯度
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)#构造实例
            _, predicted = torch.max(outputs, 1)#做预估，用max取出最大的类别

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
# acc(net)


for epoch in range(epochs):  #完整地看完一遍数据，就用一个epoch：多批次循环

    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        total += inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)

        l1_regularization, l2_regularization = torch.tensor(0.), torch.tensor(0.)
        l1_regularization = l1_regularization.to(device)
        l2_regularization = l2_regularization.to(device)

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        for param in net.parameters():
            l1_regularization += torch.sum(torch.abs(param))
        # print("loss", loss)
        # 1e5的量级
        # print("l1_regularization", l1_regularization)
        loss = loss + (1e-6) * l1_regularization 
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        # if i % 2000 == 1999:    # 每2000批次打印一次
    print('[%d] loss: %.6f' %
            (epoch + 1,  running_loss / total))
    running_loss = 0.0
    acc(net)
# acc(net)
print('Finished Training')