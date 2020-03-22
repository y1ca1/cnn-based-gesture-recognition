import utils


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', '-mode', type=str, default='train')
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--root_dir', '-sd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP')
    # parser.add_argument('--result_dir', '-rd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP')
    parser.add_argument('--result_dir', '-rd', type=str, default='/home/caiyi/PycharmProjects/gesture_MP/result')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--input_size', '-is', type=int, default=64)
    # parser.add_argument('--num_joint', '-nj', type=int, default=14)
    parser.add_argument('--fc_size', '-fc', type=int, default=2048)
    parser.add_argument('--num_class', '-nc', type=int, default=11)
    parser.add_argument('--epoch', '-epoch', type=int, default=100)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', type=float, default=100000)
    args = vars(parser.parse_args())
    utils.print_args(args)
    return args