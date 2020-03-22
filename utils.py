import numpy as np
from matplotlib import pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', '-mode', type=str, default='train')
    parser.add_argument('--gpu_id', '-id', type=str, default='1')
    parser.add_argument('--root_dir', '-sd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP')
    # parser.add_argument('--result_dir', '-rd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP')
    parser.add_argument('--result_dir', '-rd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--input_size', '-is', type=int, default=64)
    # parser.add_argument('--num_joint', '-nj', type=int, default=14)
    parser.add_argument('--fc_size', '-fc', type=int, default=2048)
    parser.add_argument('--epoch', '-epoch', type=int, default=600)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', type=float, default=100000)
    args = vars(parser.parse_args())
    print_args(args)
    return args


"""=================utility functions for training=================="""


def print_args(args):
    """ Prints the argparse argmuments applied
    Args:
      args = parser.parse_args()
    """
    max_length = max([len(k) for k, _ in args.items()])
    for k, v in args.items():
        print(' ' * (max_length - len(k)) + k + ': ' + str(v))


# GPU or CPU configuration
config = get_config()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
dtype = torch.float32


def check_accuracy_part34(loader, model, config):
    # if loader.dataset.train:
    #     print('Checking accuracy on validation set')
    # else:
    #     print('Checking accuracy on test set')
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    device = torch.device("cpu")
    dtype = torch.float32
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100 * acc


def kaiming_normal(shape, config):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32
    if len(shape) == 2:  # FC weight
        fan_in = shape[
            1]  # different from `random_weight()`, as weight for nn.Linear in pytorch is of shape: [out_feature, in_feature]
    else:
        fan_in = np.prod(shape[1:])  # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def xavier_normal(shape, config):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Xavier normalization: sqrt(2 / (fan_in + fan_out))
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32
    if len(shape) == 2:  # FC weight
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        fan_in = np.prod(shape[1:])  # conv weight [out_channel, in_channel, kH, kW]
        fan_out = shape[0] * shape[2] * shape[3]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / (fan_in + fan_out))
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        #         m.weight.data = random_weight(m.weight.size())
        #         m.weight.data = kaiming_normal(m.weight.size())
        m.weight.data = xavier_normal(m.weight.size())
        m.bias.data = zero_weight(m.bias.size())
