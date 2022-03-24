from project1_model import project1_model
import matplotlib.pyplot as plt

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar

# python main.py --lr 0.001 --optim ADADELTA
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optim', default='ADADELTA', type=str, help='optimizer')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# ===============Data================
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# ================Model================
print('==> Building model..')

net = project1_model()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# select optimizer

if args.optim == 'ADADELTA':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
elif args.optim == 'ADAGRAD':
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.optim == 'ADAM':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []
# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx == len(trainloader)-1:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss = train_loss/len(trainloader)
    train_loss_history.append(train_loss)
    train_acc = (correct/total) * 100
    train_acc_history.append(train_acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx == len(testloader)-1:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_loss = test_loss/len(testloader)
        test_loss_history.append(test_loss)
        test_acc = (correct/total) * 100
        test_acc_history.append(test_acc)

    # Save model with best accuracy.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print("Saving")
    #     model_path = './project1_model.pt'
    #     torch.save(net.state_dict(), model_path)
    #     best_acc = acc


# print number of architecture parameters
resnet_total_params = sum(p.numel()
                          for p in net.parameters() if p.requires_grad)
print("Number of trainable parameters in the model: %d\n" %
      (resnet_total_params))


for epoch in range(start_epoch, start_epoch+5):
    train(epoch)
    test(epoch)
    scheduler.step()

plt.plot(range(200), train_loss_history, '-',
         linewidth=3, label='Train error')
plt.plot(range(200), test_loss_history, '-',
         linewidth=3, label='Test error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.savefig('best_model_loss.png')
plt.clf()

plt.plot(range(200), train_acc_history, '-',
         linewidth=3, label='Train accuracy')
plt.plot(range(200), test_acc_history, '-',
         linewidth=3, label='Test accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid(True)
plt.legend()
plt.savefig('best_model_acc.png')
