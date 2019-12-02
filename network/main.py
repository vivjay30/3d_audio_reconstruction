import multiprocessing
import os

import argparse
import numpy as np
import torch
import torch.optim as optim

from d3audiorecon.network.data_loader import SpatialAudioDataset, \
    NUM_BINS
from d3audiorecon.network.train_test import train, test, \
    test_unet
from d3audiorecon.network.resnet import resnet18, resnet50
from d3audiorecon.network.simplenet import SimpleNet
from d3audiorecon.network.UNet import unet


def main(args):
    """
    Factor out common code to be used by all data corpuses.
    """
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size
    EPOCHS = 20
    LEARNING_RATE = .00001
    WEIGHT_DECAY = 0
    USE_CUDA = True
    PRINT_INTERVAL = 100
    LOG_PATH = "../data/logs/log.pkl"

    data_train = SpatialAudioDataset(args.data_train_path, task=args.task)
    data_test = SpatialAudioDataset(args.data_test_path, task=args.task)

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                              shuffle=True, **kwargs)

    # Key modifcations to resnet include changing the input and output channels
    #model = resnet50(pretrained=True, num_classes=NUM_BINS).to(device)
    #model = SimpleNet().to(device)
    #model = unet(pretrained=True).to(device)
    model = torch.load(args.model_load).to(device)
    save_prefix = "resnet" if args.task == 0 else "unet" 
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = 0

    train_losses, test_losses, test_accuracies = ([], [], [])
    # test_loss = test(model, device, test_loader)

    # test_losses.append((start_epoch, test_loss))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            #lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = train(model, device, optimizer, train_loader, None, epoch, PRINT_INTERVAL)
            if args.task == 0:
                test_loss = test(model, device, test_loader, PRINT_INTERVAL)
            elif args.task == 1:
                test_loss = test_unet(model, device, test_loader, PRINT_INTERVAL)
            train_losses.append((epoch, train_loss))
            print("Train Loss: {}".format(train_loss))
            print("Test Loss: {}".format(test_loss))
            test_losses.append((epoch, test_loss))
            
            torch.save(model, os.path.join(args.checkpoints_dir, "{}_{}.pt".format(save_prefix, epoch)))
    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()

    # torch.save(model, "trained_mixed_trim.pt")
    # finally:
    #     print('Saving final model')
    #     model.save_model(DATA_PATH + checkpoints_dir + '/%03d.pt' % epoch, 0)
    #     ep, val = zip(*train_losses)
    #     pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
    #     pt_util.plot(ep[1:], np.exp(val)[1:], 'Train perplexity', 'Epoch', 'Perplexity')
    #     ep, val = zip(*test_losses)
    #     pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
    #     pt_util.plot(ep[2:], np.exp(val)[2:], 'Test perplexity', 'Epoch', 'Perplexity')
    #     ep, val = zip(*test_accuracies)
    #     pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')
    #     return model, vocab, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for network/main.py')
    parser.add_argument("data_train_path", type=str, help="Path to training samples")
    parser.add_argument("data_test_path", type=str, help="Path to testing samples")
    parser.add_argument("task", type=int, help="0 for position prediction, 1 for source separation")
    parser.add_argument("--checkpoints-dir", type=str, help="Path to save model")
    parser.add_argument("--model-load", type=str, help="Path to starting model weights")
    parser.add_argument("--batch-size", type=int, default=2)
    main(parser.parse_args())
