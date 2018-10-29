from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from models import StupidModel, BasicBlock
from torch.optim.lr_scheduler import MultiStepLR
import time


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':
    start_epoch = 1
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

    trainset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    device = 'cuda'
    best_acc = 0
    log = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    n = '1'
    model = StupidModel()
    # model = HiResC(1)
    model = torch.nn.DataParallel(model).cuda()
    # summary((3, 32, 32), model)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-1,
                                momentum=0.9,
                                weight_decay=1e-6, nesterov=True)
    # first_scheduler = MultiStepLR(optimizer, milestones=[2], gamma=10)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60])
    cudnn.benchmark = True

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        # first_scheduler.step()
        scheduler.step()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        last_time = start_time

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

            step_time = time.time() - last_time
            last_time = time.time()
            try:
                _, term_width = os.popen('stty size', 'r').read().split()
                print(f'\r{" "*(int(term_width))}', end='')
            except ValueError:
                pass
            lss = f'{batch_idx}/{len(trainloader)} | ' + \
                  f'ETA: {format_time(step_time*(len(trainloader)-batch_idx))} - ' + \
                  f'loss: {train_loss/(batch_idx+1):.{3}} - ' + \
                  f'acc: {correct/total:.{5}}'
            print(f'\r{lss}', end='')

        print(f'\r '
              f'ToT: {format_time(time.time() - start_time)} - '
              f'loss: {train_loss/(batch_idx+1):.{3}} - '
              f'acc: {correct/total:.{5}}', end='')
        log['acc'].append(100.*correct/total)
        log['loss'].append(train_loss/(batch_idx+1))

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

        print(f' - val_acc: {correct / total:.{5}}')
        log['val_acc'].append(100.*correct/total)
        log['val_loss'].append(test_loss/(batch_idx+1))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+90):
        train(epoch)
        test(epoch)
        print(f'best: {best_acc}')
