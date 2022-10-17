
import torch
import torch.distributed as dist
import shutil
import os
import logging

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setting_learning_rate(optimizer, epoch, train_length, checkpoint, args, scheduler):
    if args.lr_policy in ['sgdr', 'sgdr_step']:
        if epoch in args.lr_custom_step or epoch == 0 or args.resume:
            index = len([x for x in args.lr_custom_step if x <= epoch])
            if index == 0:
                last = 0
            else:
                last = args.lr_custom_step[index-1]
            if index >= len(args.lr_custom_step):
                current = args.epochs
            else:
                current = args.lr_custom_step[index]
            step = current - last
            step = step * train_length

            if args.lr_policy == 'sgdr_step':
                lr = utils.adjust_learning_rate(optimizer, epoch, args)
                for group in optimizer.param_groups:
                    group['initial_lr'] = lr
                logging.info('warning: update sgdr initial_lr')

            last_epoch = -1
            if args.resume and checkpoint is not None:
                last_epoch = train_length * (epoch - last) - 1
                args.resume = False

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step, eta_min=args.eta_min, last_epoch=last_epoch)
            for group in optimizer.param_groups:
                if 'lr_constant' in group:
                    group['lr'] = group['lr_constant']
        lr_list = scheduler.get_lr()
        if isinstance(lr_list, list):
            lr = lr_list[0]
        else:
            lr = None
    elif args.lr_policy in ['cos']:
        if epoch in args.lr_custom_step or epoch == 0 or args.resume:
            index = len([x for x in args.lr_custom_step if x <= epoch])
            if index == 0:
                last = 0
            else:
                last = args.lr_custom_step[index-1]
            if index >= len(args.lr_custom_step):
                current = args.epochs
            else:
                current = args.lr_custom_step[index]
            step = current - last

            last_epoch = -1
            if args.resume and checkpoint is not None:
                last_epoch = epoch - last - 1
                args.resume = False

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step, eta_min=args.eta_min, last_epoch=last_epoch)
            for group in optimizer.param_groups:
                if 'lr_constant' in group:
                    group['lr'] = group['lr_constant']

        if epoch != 0:
            scheduler.step()

        lr_list = scheduler.get_lr()
        if isinstance(lr_list, list):
            lr = lr_list[0]
        else:
            lr = None
    else:
        lr = adjust_learning_rate(optimizer, epoch, args)
    if lr is None:
        return None, None
    return lr, scheduler

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_policy == 'decay':
        lr = args.lr * (args.lr_decay ** epoch)
    elif args.lr_policy == 'poly':
        interval = len([x for x in args.lr_custom_step if epoch >= x])
        epoch = epoch if interval == 0 else epoch - args.lr_custom_step[interval-1]
        if interval == 0:
            step = args.lr_custom_step[0]
        elif interval >= len(args.lr_custom_step):
            step = args.epochs - args.lr_custom_step[interval-1]
        else:
            step = args.lr_custom_step[interval] - args.lr_custom_step[interval-1]
        lr = args.eta_min + (args.lr - args.eta_min) * (1 - epoch * 1.0 /step)** args.lr_decay
    elif args.lr_policy == 'fix':
        lr = args.lr
    elif args.lr_policy == 'fix_step':
        lr = args.lr * (args.lr_decay ** (epoch // args.lr_fix_step))
    elif args.lr_policy in ['custom_step', 'sgdr_step']:
        interval = len([x for x in args.lr_custom_step if epoch >= x])
        lr = args.lr *(args.lr_decay ** interval)
    else:
      return None

    if optimizer is not None and args.lr_policy != 'sgdr_step':
      for param_group in optimizer.param_groups:
          if param_group.get('lr_constant', None) is not None:
              continue
          param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='meter', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, args):
    torch.save(state, args.resume_file)
    epoch = state['epoch']

    if is_best:
      if epoch > args.epochs/2:
          shutil.copy(args.resume_file, os.path.join(args.weights_dir, args.case + '-model_best.pth.tar'))
      else:
        logging.info('obtain new best accuracy, but not going to save it at epoch %d' % epoch)

    # save some special epoch
    epoch = epoch + 1
    save_epoch1 = args.lr_policy in ['sgdr', 'custom_step'] and epoch in args.lr_custom_step
    save_epoch2 = args.lr_policy in ['fix_step'] and epoch % args.lr_fix_step == 0
    save_epoch3 = args.save_freq >= 1 and epoch % args.save_freq == (args.save_freq - 1)
    save_epoch4 = epoch == args.epochs
    save_epoch5 = epoch == args.stable_epoch
    save_epoch6 = epoch == (args.stable_epoch + args.extra_epoch)
    if save_epoch1 or save_epoch2 or save_epoch3 or save_epoch4 or save_epoch5 or save_epoch6:
        epoch = state['epoch']
        shutil.copy(args.resume_file, os.path.join(args.weights_dir, args.case + '-epoch' + str(epoch) + '-checkpoint.pth.tar'))

def load_state_dict(model, state_dict, resume_scope='', unresume_scope='', verbose=False, logger=None):
    if logger is None:
      logger = logging

    checkpoint = state_dict
    # remove the module in the parallel model
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict
    checkpoint_disk = checkpoint

    if resume_scope != '':
        pretrained_dict = {}
        for k, v in list(checkpoint.items()):
            for resume_key in resume_scope.split(','):
                if resume_key in k:
                    pretrained_dict[k] = v
                    break
        checkpoint = pretrained_dict

    if unresume_scope != '':
        pretrained_dict = {}
        unresume_set = set(unresume_scope.split(','))
        for k, v in list(checkpoint.items()):
            origin = set(k.split('.'))
            merge = origin - unresume_set
            if len(merge) == len(origin):
                pretrained_dict[k] = v
            else:
                module = model
                for i in k.rsplit('.', 1)[0].split('.'):
                    module = getattr(module, i, None)
                    if module is None:
                        break
                if module is not None and hasattr(module, 'import_pretrain'):
                    module.import_pretrain(k, v.cpu())
        checkpoint = pretrained_dict

    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model.state_dict():
            shape = model.state_dict()[k].shape
            if v.shape != shape:
                logger.info("=> warning: {} shape mismatch {} vs {}".format(k, v.shape, shape))
                try:
                    v = v.reshape(shape)
                except RuntimeError as e:
                    logger.info("=> warning: skip recovering {}".format(k))
                    continue
            pretrained_dict[k] = v

    checkpoint = model.state_dict()

    unresume_dict = set(checkpoint) - set(pretrained_dict)
    unused_dict = set(checkpoint_disk) - set(pretrained_dict)
    if len(unresume_dict) != 0:
        logger.info("=> UNResume weigths:")
        for item in sorted(unresume_dict):
            logger.info('-> %r' % item)
        if verbose:
            print("=> UNResume weigths:")
            for item in sorted(unresume_dict):
                print('%r' % item)

    if len(unused_dict) != 0:
        logger.info("=> UNUsed weigths:")
        for item in sorted(unused_dict):
            logger.info("-> %r" % item)
        if verbose:
            print("=> UNUsed weigths:")
            for item in sorted(unused_dict):
                print("%r" % item)

    checkpoint.update(pretrained_dict)

    return model.load_state_dict(checkpoint)

def import_state_dict(old, new, mapping=None, raw=False, raw_prefix=''):
    if torch.cuda.is_available():
        disk_checkpoint = torch.load(old)
    else:  # force cpu mode
        disk_checkpoint = torch.load(old, map_location='cpu')

    checkpoint = None
    if raw:
        checkpoint = disk_checkpoint

    for k, v in disk_checkpoint.items():
      if type(v) in [ int, float, str ]:
          print("resuming %s := %r" %(k, v))
      else:
          print("resuming %s" % k)
      if k in [ "state_dict", "model"]:
          checkpoint = v

    if checkpoint is None:
        print("resuming state_dict := None")
        return

    pretrained_dict = dict()
    if mapping != None and type(mapping) is dict:
        for k, v in mapping.items():
            for name, value in checkpoint.items():
                if k in name:
                  pretrained_dict[name.replace(k, v)] = value
                  print("mapping %s to %s" % (name, name.replace(k, v)))
                else:
                  pretrained_dict[name] = value
        checkpoint = pretrained_dict

    if raw:
        disk_checkpoint = dict()
        disk_checkpoint['state_dict'] = checkpoint
    else:
        disk_checkpoint['state_dict'] = checkpoint
    if old != new and new != "":
        torch.save(disk_checkpoint, new)
        print("save to %s" % new)

    if raw and new != "":
        if raw_prefix != "":
            pretrained_dict = dict()
            for name, value in checkpoint.items():
                pretrained_dict[raw_prefix + name] = value
            checkpoint = pretrained_dict
        torch.save(checkpoint, new + '.raw')
        print("save raw to %s" % new + '.raw')

def check_folder(folder):
  if not os.path.exists(folder):
    os.mkdir(folder)

def check_file(fname):
  return os.path.isfile(fname)

def custom_state(model):
    for m in model.modules():
        if hasattr(m, 'freeze_bn'):
            m.eval()
            if hasattr(m, 'freeze_bn_affine'):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def check_pid(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False # no process with pid
    else:
        return True

def load_pretrained(model, args, logger=None):
    if logger is None:
        logger = logging

    if check_file(args.pretrained):
        logger.info("load pretrained from %s" % args.pretrained)
        if torch.cuda.is_available():
            checkpoint = torch.load(args.pretrained)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
        # logger.info("load pretrained ==> last epoch: %d" % checkpoint.get('epoch', 0))
        # logger.info("load pretrained ==> last best_acc: %f" % checkpoint.get('best_acc', 0))
        # logger.info("load pretrained ==> last learning_rate: %f" % checkpoint.get('learning_rate', 0))
        #if 'learning_rate' in checkpoint:
        #    lr = checkpoint['learning_rate']
        #    logger.info("resuming ==> learning_rate: %f" % lr)
        try:
            load_state_dict(model, checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        except RuntimeError as err:
            logger.info("Loading pretrained model failed %r" % err)
    else:
        logger.info("no pretrained file exists({}), init model with default initlizer".
            format(args.pretrained))

