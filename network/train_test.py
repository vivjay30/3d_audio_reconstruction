import os
import json

import tqdm
import torch
import torch.nn as nn

import numpy as np

import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pylab


def save_spectogram(input_spectogram, output_filename):
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(librosa.power_to_db(input_spectogram, ref=np.max))
    pylab.savefig(output_filename, bbox_inches=None, pad_inches=0)
    pylab.close()

def train(model, device, optimizer, train_loader, lr, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches. 
        # Otherwise the backward would try to go all the way to the beginning every time.
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(-1)[1]
        loss = nn.MSELoss(reduction="mean")(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, direction, mask) in enumerate(test_loader):
            data, direction, mask = data.to(device), direction.to(device), mask.to(device)
            masknp = mask.numpy()[0, 0, :, :]
            datanp = data.numpy()[0, 0, :, :]
            save_spectogram(datanp, "mixed0.png")
            import pdb
            pdb.set_trace()
            masknp = np.clip(masknp, 0, 1.0)
            import cv2
            cv2.imwrite("mask.png", (masknp * 255).astype(np.uint8))            
            import pdb
            pdb.set_trace()
            output = model(data)
            loss = nn.MSELoss(reduction="mean")(output, label).item()
            test_loss += loss
            # Comment this out to avoid printing test results
            print("GT: {} \nPred: {} \nLoss: {}".format(label, output, loss))

    test_loss /= len(test_loader)

    return test_loss