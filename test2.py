import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from project1_model import project1_model
from utils import progress_bar


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './project1_model.pt'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['net'], strict=False)

best_acc = checkpoint['acc']
print(best_acc)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

test_loss_history = []


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


test(5)

# model.eval()
# corrects = 0
# for batch_idx, (inputs, labels) in enumerate(testloader, 1):
#     with torch.set_grad_enabled(False):
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#     corrects += torch.sum(preds == labels.data)
# print(corrects.float() / len(testloader))


# pred = predicted_output.data.max(1)[1] # get the index of the max log-probability
#       test_corrects += pred.eq(labels.data).cpu().sum()
#       test_accuracy = 100. * test_corrects / len(testDataLoader.dataset)
