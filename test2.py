import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from project1_model import project1_model
from utils import progress_bar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './project1_model.pt'
model.load_state_dict(torch.load(
    model_path, map_location=device), strict=False)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

model.eval()
corrects = 0
total = 0
for batch_idx, (inputs, labels) in enumerate(testloader, 1):
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        corrects += predicted.eq(labels).sum().item()
test_accuracy = 100.*corrects/total
print(test_accuracy)
