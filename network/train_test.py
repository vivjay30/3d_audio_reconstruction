import os
import json

import tqdm
import torch
import torch.nn as nn

import cv2
import numpy as np


def train(model, device, optimizer, train_loader, lr, epoch, log_interval=20):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        data = (data - data.mean()) / (data.std() + 1e-8)
        # Separates the hidden state across batches. 
        # Otherwise the backward would try to go all the way to the beginning every time.
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # data = (output > 0.5).cpu().numpy()
            # cv2.imwrite("sample_mask_train.png", (output.cpu().detach().numpy()[0, 0, :, :] * 255).astype(np.uint8))
            # Comment this out to avoid printing test results
            print("Loss: {}".format(loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return np.mean(losses)


def test(model, device, test_loader, log_interval=100):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            data = (data - data.mean()) / (data.std() + 1e-8)
            output = model(data)
            loss = model.loss(output, label)
            test_loss += loss
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct

            if batch_idx % log_interval == 0:
                # Comment this out to avoid printing test results
                print("GT: {} \nPred: {} \nLoss: {}".format(label, pred, loss))

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    return test_loss

def test_unet(model, device, test_loader, log_interval=20):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            data = (data - data.mean()) / (data.std() + 1e-8)
            output = model(data)
            loss = model.loss(output, label)
            test_loss += loss

            if batch_idx % log_interval == 0:
                data = (output > 0.5).cpu().numpy()
                cv2.imwrite("sample_mask_test.png", (output.cpu().detach().numpy()[0, 0, :, :] * 255).astype(np.uint8))
                # Comment this out to avoid printing test results
                print("Loss: {}".format(loss))

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f})\n'.format(
        test_loss))

    return test_loss