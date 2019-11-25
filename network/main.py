import multiprocessing

import argparse
import numpy as np
import torch
import torch.optim as optim

from d3audiorecon.network.data_loader import SpatialAudioDataset, \
    NUM_BINS
from d3audiorecon.network.train_test import train, test
from d3audiorecon.network.resnet import resnet18, resnet50
from d3audiorecon.network.simplenet import SimpleNet

DATA_TRAIN_PATH = "../data/output_sounds/train_iso/"
DATA_TEST_PATH = "../data/output_sounds/test_iso/"

def main():
    """
    Factor out common code to be used by all data corpuses.
    """
    BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = .00001
    WEIGHT_DECAY = 0 #0.0005
    USE_CUDA = True
    PRINT_INTERVAL = 10
    LOG_PATH = "../data/logs/log.pkl"

    data_train = SpatialAudioDataset(DATA_TRAIN_PATH)
    data_test = SpatialAudioDataset(DATA_TEST_PATH)

    checkpoints_dir = "../data/checkpoints"

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
    model = torch.load("trained_mixed.pt")
    #model = SimpleNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = 0

    train_losses, test_losses, test_accuracies = ([], [], [])
    # test_loss = test(model, device, test_loader)

    # test_losses.append((start_epoch, test_loss))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            #lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = train(model, device, optimizer, train_loader, None, epoch, PRINT_INTERVAL)
            test_loss = test(model, device, test_loader)
            train_losses.append((epoch, train_loss))
            print("Train Loss: {}".format(train_loss))
            # print("Test Loss: {}".format(test_loss))
            test_losses.append((epoch, test_loss))

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()

    torch.save(model, "trained_mixed.pt")
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
    parser.add_argument("--train_path")
    parser.add_argument("--test_path")

    args = parser.parse_args()
    if (args.train_path):
        DATA_TRAIN_PATH=args.train_path
    if (args.test_path):
        DATA_TEST_PATH=args.test_path

    print("loading train files in ", DATA_TRAIN_PATH)
    print("loading test files in ", DATA_TEST_PATH)
    main()