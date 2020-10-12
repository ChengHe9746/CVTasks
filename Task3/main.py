# -*- coding: utf-8 -*-

import torch.optim as optim
import torch
import argparse
import numpy as np

import model
import data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Experiment')
    parser.add_argument('--wd', default=0.0001, help='weight decay')
    parser.add_argument('--batch_size', default=128, help='batch size')
    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--epochs', default=10, help='number of epochs to run')
    parser.add_argument('--data_root', default='/mnist_data', help='root dir of data')
    parser.add_argument('--seed', default=44, help='random seed')
    parser.add_argument('--decreasing_lr', default=[80, 120], help='lr decay')
    parser.add_argument('--test_interval', default=1, help='test interval epoch')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda: 0" if USE_CUDA else "cpu")
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)

    # data loader
    train_loader, test_loader = data.get_data(batch_size=args.batch_size, root=args.data_root, num_workers=1)

    # model
    lenet_5 = model.LeNet5()
    lenet_5 = lenet_5.to(device)

    # SGD
    optimizer = optim.SGD(lenet_5.parameters(), lr=args.lr, momentum=0.9)

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    train_accuracy = []
    test_accuracy = []

    # train
    best_acc = 0.0
    for epoch in range(args.epochs):
        print('-'*20)
        lenet_5.train()
        if epoch in args.decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        epoch_training_loss = 0.0
        num_batches = 0
        accuracy = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = lenet_5(data)

            loss = loss_func(output, target)

            loss.backward()

            optimizer.step()

            epoch_training_loss += loss.item()
            num_batches += 1

            # calculate train accuracy
            actual_val = target.clone().cpu().numpy()
            pred_val = output.cpu().detach().numpy()
            pred_val = np.argmax(pred_val, axis=1)
            tmp = actual_val == pred_val
            accuracy += tmp.sum() / tmp.shape[0]
        train_accuracy.append(accuracy / num_batches)

        print('epoch: %d, loss: %0.4f, train acc: %0.4f %%' %
              (epoch, epoch_training_loss / num_batches, accuracy / num_batches * 100))

        if epoch % args.test_interval == 0:
            lenet_5.eval()
            correct = 0
            tes_num_batches = 0
            for data, target in test_loader:
                tes_num_batches += 1
                data = data.to(device)
                pred_val = lenet_5(data)
                pred_val = pred_val.cpu().detach().numpy()
                pred_val = np.argmax(pred_val, axis=1)
                target = target.numpy()
                tmp = target == pred_val
                correct += tmp.sum() / tmp.shape[0]
            test_acc = correct / tes_num_batches
            print('epoch: %d, test dataset accuracy: %0.4f %%' % (epoch, test_acc * 100))
            if test_acc > best_acc:
                best_acc = test_acc

    # test model
    correct = 0.0
    num_batches = 0
    lenet_5.eval()
    for test_data in test_loader:
        num_batches += 1
        inputs, actual_val = test_data
        inputs = inputs.to(device)
        pred_val = lenet_5(inputs)
        pred_val = pred_val.cpu().detach().numpy()
        pred_val = np.argmax(pred_val, axis=1)
        actual_val = actual_val.numpy()
        tmp = actual_val == pred_val
        correct += tmp.sum() / tmp.shape[0]
    print('Final Test dataset accuracy: %0.4f %%' % (correct / num_batches * 100))

    print('Best Test dataset accuracy: %0.4f %%' % (best_acc * 100))


















