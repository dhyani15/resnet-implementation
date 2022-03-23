import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from project1_model import project1_model
from utils import progress_bar

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = project1_model().to(device)
    model_path = './project1_model.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=False)

    best_acc = checkpoint['acc']
    print(best_acc)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=0.1, weight_decay=5e-4)

    train_loss_history = []
    test_loss_history = []

    model.eval()
    corrects = 0
    for batch_idx, data in enumerate(testloader, 1):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        print(corrects.float() / len(testloader.dataset))

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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

    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
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

    for epoch in range(5):
        train(epoch)
        test(epoch)
