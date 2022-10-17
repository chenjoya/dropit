

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Pytorch training")
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--root', type=str, default='/data/imagenet/', help='dataset root')
    parser.add_argument('--model', '-m', '--arch', default='resnet18', type=str)

    parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
    parser.add_argument('--addition_augment', '--aa', action='store_true', default=False)
    parser.add_argument('--workers', '-j', default=12, type=int, metavar='HP')
    parser.add_argument('--iter-size', default=1, type=int)
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size')
    parser.add_argument('--val_batch_size', '-v', type=int, default=50, help='test batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate')
    parser.add_argument('--lr_policy', type=str, default='decay', help='learning rate update policy')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='decay for every epoch')
    parser.add_argument('--eta_min', type=float, default=0, help='min lr for sgdr')
    parser.add_argument('--lr_fix_step', type=int, default=30, help='learning rate step for fix_step')
    parser.add_argument('--lr_custom_step', type=str, default='20,30,40', help='learning rate steps for custom_step')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', "--wd", type=float, default=0.0001, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no_decay_small', default=True, action='store_true')
    parser.add_argument('--decay_small', default=False, action='store_true')
    parser.add_argument('--grad_clip', type=float, default=None, help='gradient clipping')
    parser.add_argument('--save_freq', '-s', default=-1, type=int, help='epoch to save model (default: -1)')
    parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'ADAM'])
    parser.add_argument('--device-ids', '-g', default=[0,1,2,3], type=int, nargs='+')

    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--world_size', type=int, default=1)

    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--opt_level', default='O0', type=str)

    parser.add_argument('--wakeup', default=0, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--stable', default=0, type=int)
    parser.add_argument('--stable_epoch', default=0, type=int)
    parser.add_argument('--warmup_epoch', default=0, type=int)
    parser.add_argument('--extra_epoch', default=0, type=int)
    parser.add_argument('--delay', default=0, type=float)

    parser.add_argument('--evaluate', '-e', action='store_true', default=False)
    parser.add_argument('--pretrained', dest='pretrained', default="", type=str)
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume training')
    parser.add_argument('--resume_file', type=str, default='checkpoint.pth.tar')
    parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
    parser.add_argument('--log_dir', type=str, default='exp', help='experiment name')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    # KD
    parser.add_argument('--distill_teacher', type=str, default='', help='teacher model used for KD')
    parser.add_argument('--distill_loss_alpha', type=float, default=0.1, help='loss alpha used for KD')
    parser.add_argument('--distill_loss_temperature', type=float, default=5, help='loss temperature used for KD')
    parser.add_argument('--distill_loss_type', type=str, default='soft', help='loss type for KD')

    # repvgg
    parser.add_argument('--repvgg_block', type=str, default='', help='repvgg blocks')

    parser.add_argument('--case', type=str, default='official', help='identify the configuration of the training')
    parser.add_argument('--keyword', default='pretrain', type=str, help='key features')
    return parser

def get_config():
    parser = get_parser()
    args = parser.parse_args()

    if isinstance(args.lr_custom_step, str):
        args.lr_custom_step = [int(x) for x in args.lr_custom_step.split(',')]
    if isinstance(args.keyword, str):
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    return args

