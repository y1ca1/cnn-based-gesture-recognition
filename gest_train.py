import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import time
import sys
sys.path.append('..')
import utils
from GestNet_UoG import ImageFolderSplitter, DatasetFromFilename
from gest_model import ThreeLayerConvNet, TwoLayerConvNet
from our_parser import get_config
print_every = 100


def train(config):
    """
    Train a REN model on our datasets

    Inputs:
    - config, a dictionary containing necessary parameters

    Returns: Nothing, but prints model accuracies during training and testing.
    """
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32

    # aug = Augmentation(shift_limit=0.01, scale_limit=0.2, rotate_limit=180, apply_prob=1.0)
    # train_dataset = RENDataset(mode='train', store_dir='/hand_pose_data/nyu', transform=aug, cube=(300, 300, 300))
    # train_generator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)
    #
    # test_dataset = RENDataset(mode='test', store_dir='/hand_pose_data/nyu', transform=None, cube=(300, 300, 300))
    # test_generator = DataLoader(test_dataset, batch_size=config['batch_size']*4, shuffle=False, num_workers=16)
    batch_size = config['batch_size']

    train_transforms = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    val_transforms = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    store_dir = '%s/%s' % (config['root_dir'], 'dataset/data.h5')
    assert os.path.exists(store_dir)
    splitter = ImageFolderSplitter(store_dir)

    x_train, y_train = splitter.getTrainingDataset()
    training_dataset = DatasetFromFilename(x_train, y_train, transforms=train_transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    x_valid, y_valid = splitter.getValidationDataset()
    validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=val_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # initialize model and optimizer
    model = TwoLayerConvNet(config)
    optimizer = optim.Adam(model.parameters(), lr=config['lr_start'], betas=(0.9, 0.999), eps=1e-07)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_func = nn.CrossEntropyLoss()

    # save model and results
    model_dir = '%s/TwoLayerConv/model.pkl' % config['result_dir']
    # writer = SummaryWriter(log_dir='%s/depth_96' % config['result_dir'])

    # training
    best_loss, iter = 100, 0
    for epoch in range(config['epoch']):
        print('==========================================Epoch %i============================================' % epoch)

        # test
        print('--------------------------------------- Testing --------------------------------------')
        start_time = time.time()
        test_loss, test_acc_history = [], []
        num_correct = 0
        num_samples = 0
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for t, (x, y) in enumerate(validation_dataloader):
                model.eval()  # dropout layers will work in eval mode
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                t_loss = loss_func(scores, y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            test_acc = 100 * float(num_correct) / num_samples
            # if t % print_every == 0:
            # print('Iteration %d, loss = %.4f' % (t, t_loss.item()))
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, test_acc))
            # t_loss = F.cross_entropy(scores, y)
            test_loss.append(t_loss.detach().cpu().numpy())
            test_acc_history.append(test_acc)
            if best_loss > np.mean(test_loss):
                torch.save(model.state_dict(), model_dir)
                best_loss = np.mean(test_loss)
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))
                # model.load_state_dict(torch.load(model_dir))

        end_time = time.time()
        # print('>>> [epoch {:2d} Training loss: {:.4f}, Accuracy {:.4f}, lr: {:.6f},\n'
        #       'time used: {:.2f}s.'
        #       .format(epoch, np.mean(test_loss), test_acc_history,
        #               scheduler.get_lr()[0], end_time - start_time))

        # train
        print('--------------------------------------- Training --------------------------------------')
        start_time = time.time()
        train_loss, acc_history = [], []
        for t, (x, y) in enumerate(training_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            # loss = F.cross_entropy(scores, y)
            loss = loss_func(scores, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            scheduler.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc = utils.check_accuracy_part34(validation_dataloader, model, config)
            train_loss.append(loss.detach().cpu().numpy())
            acc_history.append(acc)

        end_time = time.time()
        # print('>>> [epoch {:2d}/ iter {:6d}] Training loss: {:.4f}, Accuracy {:.4f}, lr: {:.6f},\n'
        #       'time used: {:.2f}s.'
        #       .format(epoch, iter, np.mean(train_loss), acc_history,
        #               scheduler.get_lr()[0], end_time - start_time))


def test(config):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    device = torch.device("cpu")
    dtype = torch.float32

    # model_dir = '%s/ThreeLayerConv/model.pkl' % config['result_dir']
    store_dir = '%s/%s' % (config['root_dir'], 'dataset/data.h5')
    assert os.path.exists(store_dir)
    splitter = ImageFolderSplitter(store_dir)
    val_transforms = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    x_valid, y_valid = splitter.getValidationDataset()
    validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=val_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True)

    # initialize model and optimizer
    # initialize model and optimizer
    model = ThreeLayerConvNet(config)
    # model.load_state_dict(torch.load(model_dir))
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # load trained model
    model_dir = '%s/ThreeLayerConv/model.pkl' % config['result_dir']
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
        print('Model is loaded from %s' % model_dir)
    else:
        print('[Error] cannot find model file.')

    print('--------------------------------------- Testing --------------------------------------')
    while True:
        start_time = time.time()
        test_loss, test_acc_history = [], []
        test_acc = utils.check_accuracy_part34(validation_dataloader, model, config)
        test_acc_history.append(test_acc)

        end_time = time.time()
    # print('>>> Accuracy {:.4f}, \n'
    #       'time used: {:.2f}s.'
    #       .format(test_acc_history,
    #               end_time - start_time))


def main():
    config = get_config()
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # os.environ["OMP_NUM_THREADS"] = "1"
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'test':
        test(config)
    else:
        raise ValueError('mode %s errors' % config['mode'])


if __name__ == '__main__':
    main()
