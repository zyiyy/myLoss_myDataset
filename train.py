# coding=utf-8


from LinearRegression import LinearRegression
from loss_function import myLoss
from load_dataset import train_loader, batch_size
from torch import optim
from torch.autograd import Variable
import torch


model = LinearRegression()
criterion = myLoss(w1=0.3, w2=0.7, w3=5)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

num_epochs = 1000
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.numpy(), labels.numpy()
        inputs = Variable(torch.FloatTensor(inputs))
        labels = Variable(torch.FloatTensor(labels))
        inputs = inputs.view(batch_size, -1)
        labels = labels.view(batch_size, -1)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels, model.named_parameters())

        # backward
        optimizer.zero_grad()   # 梯度归零
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数

        if (epoch + 1) % 20 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))

        # print(model.state_dict())

        # 保存参数
        torch.save({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict()
        }, 'checkpoint')


