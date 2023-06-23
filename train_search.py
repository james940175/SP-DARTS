import os
import sys
sys.path.insert(0, '../')
import time
import glob
import numpy as np
import torch
import shutil
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random

from search_model import TinyNetworkDarts
from cell_operations import SearchSpaceNames
from architect import Architect
from copy import deepcopy
from nas_201_api import NASBench201API as API

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120'], help='choose dataset')
parser.add_argument('--search_space', type=str, default='nas-bench-201')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--ckpt_interval', type=int, default=10, help='frequency for ckpting')
parser.add_argument('--proj_crit', type=str, default='acc', choices=['loss', 'acc'], help='criteria for projection')
####sp-nas hyperparameters
parser.add_argument('--prune_op_num', type=int, default=4, help='number of operation pruned in each epoch each stage')
parser.add_argument('--prune_epochs', type=int, default=30, help='epochs in each stage ')
parser.add_argument('--warmup', type=int, default=20, help='warmup only weights stage epochs')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate of skip connect')
parser.add_argument('--infer_test_portion', type=float, default=0.5, help='portion of test operation strength data')
parser.add_argument('--method', type=str, default='SP-DARTS', choices=['SP-DARTS', 'SRP-DARTS'], help='choose method')

args = parser.parse_args()


expid = args.save
if args.method == 'SP-DARTS':
    args.save = './experiments-SP-DARTS/nasbench201/search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed)
else:
    args.save = './experiments-SRP-DARTS/nasbench201/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save += '-' + args.dataset
scripts_to_save = glob.glob('*.py')

utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)

fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10
    
def set_random_seed(seed):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    #torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    np.random.seed(args.seed)
    gpu = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)
    
    
    
    api = API('../NAS-Bench-201-v1_1-096897.pth')


    #### model
    criterion = nn.CrossEntropyLoss()
    search_space = SearchSpaceNames[args.search_space]
    model = TinyNetworkDarts(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, dropout=args.dropout, args=args)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    architect = Architect(model, args)


    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)
    

    epochs = args.warmup + args.prune_epochs
    logging.info(' ')
    logging.info('Epochs: %d', epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, float(epochs), eta_min=args.learning_rate_min)

    #### training
    for epoch in range(epochs):
        lr = scheduler.get_lr()[0]
        ## data aug
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)

        if args.proj_crit == 'loss':
            crit_idx = 1
        if args.proj_crit == 'acc':
            crit_idx = 0
        
        ##mask operations
        if epoch >= args.warmup:
            logging.info('Mask before: ')
        logging.info(model._arch_parameters)
        logging.info(' ')

        if epoch >= args.warmup:
            with torch.no_grad():
                m = Mask(model._arch_parameters)
                m.prune_op = []

                validation_array = []

                for i in range(len(model._arch_parameters)):
                    selected_eid = i
                    temp = []
                    logging.info(' ')
                    logging.info('Masking %d Edge', i+1)
                    logging.info('##################################################################')
                    for opid in range(model.num_op):
                        logging.info(' ')
                        logging.info('Masking %d Operation', opid+1)
                        weights = model._arch_parameters.clone().detach()
                        proj_mask = torch.ones_like(weights[selected_eid])
                        proj_mask[opid] = 0
                        weights[selected_eid] = weights[selected_eid] * proj_mask

                        valid_stats = infer_test(valid_queue, model, criterion, log=False, eval=False, weights=weights)
                        crit = valid_stats[crit_idx]

                        temp.append(crit.item())

                        logging.info('valid_acc %f', valid_stats[0])
                        logging.info('valid_loss %f', valid_stats[1])
                    validation_array.append(temp)

                logging.info(' ')
                logging.info(validation_array)

                for i in range(len(validation_array)):
                    min_array = []
                    for j in range(args.prune_op_num):
                        min = np.argmax(validation_array[i])
                        min_array.append(min)
                        validation_array[i][min] = 0
                    m.prune_op.append(min_array)
            
            logging.info(' ')
            logging.info(' ')
            logging.info('Masking operations: ')
            logging.info(m.prune_op)
            logging.info(' ')
            logging.info(' ')
            
            if args.method == 'SP-DARTS':
                m.do_mask_spdarts()
            else:
                m.do_mask_srpdarts(epoch)

            model._arch_parameters.data.copy_(m.alpha_value)
            logging.info('Mask after')
            logging.info(model._arch_parameters)

        ## training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, model.optimizer, lr, epoch)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        ## eval
        if epochs - epoch < 5:
            valid_acc, valid_obj = infer(valid_queue, model, criterion, log=True, weights=model.get_mask_weights())
            logging.info('valid_acc  %f', valid_acc)
            logging.info('valid_loss %f', valid_obj)

        #### scheduling
        scheduler.step()

        #### saving
        save_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'alpha': model.arch_parameters(),
            'optimizer': model.optimizer.state_dict(),
            'arch_optimizer': architect.optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if save_state['epoch'] % args.ckpt_interval == 0:
            utils.save_checkpoint(save_state, False, args.save, per_epoch=True)

    
    ## select the best operation
    logging.info(' ')
    logging.info(' ')
    logging.info(' ')
    logging.info(' ')
    logging.info(' ')
    logging.info('Starting to select the best operations..... ')

    best_op_array = []
    final_alpha_value = model._arch_parameters.data.cpu().numpy()
    
    logging.info(' ')
    logging.info(' ')
    logging.info(final_alpha_value)
    logging.info(' ')

    validation_array = []

    for i in range(len(model._arch_parameters)):
        selected_eid = i
        temp = []
        logging.info(' ')
        logging.info('Masking %d Edge', i+1)
        logging.info('##################################################################')
        for opid in range(model.num_op):
            logging.info(' ')
            logging.info('Masking %d Operation', opid+1)
            weights = model._arch_parameters.clone().detach()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            valid_stats = infer(valid_queue, model, criterion, log=False, eval=False, weights=weights)
            crit = valid_stats[crit_idx]
            temp.append(crit.item())

            logging.info('valid_acc %f', valid_stats[0])
            logging.info('valid_loss %f', valid_stats[1])
        validation_array.append(temp)

    logging.info(' ')
    logging.info('Validation Array')
    logging.info(validation_array)
    logging.info(' ')

    for i in range(len(validation_array)):
        best_op_array.append(np.argmin(validation_array[i]))

    logging.info('The Best Operation Combination: ')
    logging.info(' ')
    logging.info(best_op_array)
    logging.info(' ')
    logging.info(' ')
    logging.info(' ')

    op_array= ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    final_op_combination = '|' + op_array[best_op_array[0]] + '~0|+|' + op_array[best_op_array[1]] + '~0|' + op_array[best_op_array[2]] + '~1|+|' + op_array[best_op_array[3]] + '~0|' + op_array[best_op_array[4]] + '~1|' + op_array[best_op_array[5]] + '~2|'
    logging.info(final_op_combination)
    logging.info(' ')
    logging.info(' ')
    logging.info(' ')

    #result = api.query_by_arch(model.genotype(), hp='200')
    result = api.query_by_arch(final_op_combination, hp='200')
    logging.info('{:}'.format(result))


    


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()


    for step in range(len(train_queue)):
        model.train()

        ## data
        input, target = next(iter(train_queue))
        input = input.cuda(); target = target.cuda(non_blocking=True)
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda(); target_search = target_search.cuda(non_blocking=True)

        ## train alpha
        shared = None
        if epoch >= args.warmup:
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            shared = architect.step(input, target, input_search, target_search,
                                eta=lr, network_optimizer=optimizer)

        ## train weight
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args, shared=shared)

        ## logging
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,
          log=True, eval=True, weights=None, double=False, bn_est=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval() if eval else model.train()
    
    if bn_est:
        _data_loader = deepcopy(valid_queue)
        for step, (input, target) in enumerate(_data_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
        model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if double:
                input = input.double(); target = target.double()
            
            logits = model(input) if weights is None else model(input, weights=weights)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer_test(valid_queue, model, criterion,
          log=True, eval=True, weights=None, double=False, bn_est=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval() if eval else model.train()
    
    if bn_est:
        _data_loader = deepcopy(valid_queue)
        for step, (input, target) in enumerate(_data_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
        model.eval()

    temp = -1
    sample_list = random.sample(range(len(valid_queue)), int(len(valid_queue)*args.infer_test_portion))

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):

            temp+=1
            if(temp not in sample_list):
                continue

            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if double:
                input = input.double(); target = target.double()
            
            logits = model(input) if weights is None else model(input, weights=weights)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        
    return top1.avg, objs.avg


class Mask:
    def __init__(self, alpha_value):
        self.alpha_value = alpha_value
        self.prune_op = []
    
    def print_alpha(self):
        logging.info(self.alpha_value)

    def do_mask_spdarts(self):
        proj_mask = torch.ones_like(self.alpha_value).cuda()
        for i in range(len(self.prune_op)):
            for j in range(len(self.prune_op[i])):
                    proj_mask[i][self.prune_op[i][j]] = 0
        self.alpha_value = self.alpha_value * proj_mask
        #logging.info('mask done')

    def do_mask_srpdarts(self, epoch):
        proj_mask = torch.ones_like(self.alpha_value).cuda()
        if args.prune_epochs - 1 == 0:
            mask_rate = 0.0
        else:
            mask_rate = 1 - ((epoch-args.warmup)/(args.prune_epochs-1))
        logging.info('Mask rate: %f', mask_rate)

        for i in range(len(self.prune_op)):
            for j in range(len(self.prune_op[i])):
                    proj_mask[i][self.prune_op[i][j]] = mask_rate
        self.alpha_value = self.alpha_value * proj_mask
        #logging.info('mask done')


if __name__ == '__main__':
    main()