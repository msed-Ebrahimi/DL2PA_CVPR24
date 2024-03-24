import os
import argparse
import random
import time
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch
import pprint
from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter, accuracy
import warnings
from backbone.balanced.cifar100 import resnet as resnet32_balancedC100
from backbone.balanced.imagenet200 import resnet as resnet32_balancedIN200
import backbone.LT.resnet as resnet_LT
from backbone.balanced.imagenet import network as largenet
import backbone.LT.resnetIN as resnet_IN
from backbone.classifiers import fixed
from backbone.classifiers import learnable
from utils import dataset, calibration,save_checkpoint
from utils import LT_utils, B_utils
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args

def train(train_loader, model, classifier, criterion, optimizer, epoch, config, logger,scheduler=None,scaler=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    classifier.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        if i > end_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)
        labels = target.clone()

        '''begin long tail'''
        if config.dataset.endswith('LT'):
            if config.fixed_classifier:
                # weighted by the inverse ratio of the number of samples per class
                learned_norm = LT_utils.produce_Ew(target, config.num_classes)
                if config.dataset == 'imagenetLT':
                    WP = learned_norm * classifier.module.polars
                else:
                    WP = learned_norm * classifier.polars

                # using mixup augmentation
                images, targets_a, targets_b, lam = LT_utils.mixup_data(images, target, alpha=config.alpha)
                feat = model(images)
                feat = classifier(feat)
                classifier.forward_momentum(feat.detach(), labels.detach())
                output = classifier.predictLT(feat, WP)

                loss_a = LT_utils.LTloss(feat=feat, target=WP[:, targets_a].T, reg_lam=config.reg_lam)
                loss_b = LT_utils.LTloss(feat=feat, target=WP[:, targets_b].T, reg_lam=config.reg_lam)
                loss = lam * loss_a + (1 - lam) * loss_b

            else:
                feat = model(images)
                output = classifier(feat)
                loss = LT_utils.mixup_criterion(criterion, output, targets_a, targets_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''end long tail'''
        else:
            if config.dataset == 'imagenet':
                _, target = target.topk(1, 1, True, True)
                labels = target
                with autocast():
                    P = classifier.module.polars[:, target].T
                    feat = model(images)
                    if config.fixed_classifier:
                        classifier.module.forward_momentum(feat.detach(), target.squeeze(1).detach())
                        loss = B_utils.BLoss(criterion, feat, P)
                        output = classifier.module.predict(feat)
                    else:
                        output = classifier(feat)
                        loss = criterion(output, target)

                # compute gradient and do SGD step
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # adjust learning rate
                if scheduler is not None:
                    scheduler.step()
            else:
                P = classifier.polars[:, target].T
                feat = model(images)
                if config.fixed_classifier:
                    classifier.forward_momentum(feat.detach(), labels.detach())
                    loss = B_utils.BLoss(criterion, feat, P)
                    output = classifier.predict(feat)
                else:
                    output = classifier(feat)
                    loss = criterion(output, target)
                # compute gradient and do SGD step
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        acc1, acc5 = accuracy(output, labels.cuda(config.gpu, non_blocking=True), topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)

    if scheduler is not None:
        scheduler.step()

def validate(val_loader, model, classifier, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            if config.dataset.endswith('LT'):
                if config.fixed_classifier:
                    feat = model(images)
                    feat = classifier(feat)
                    output = classifier.predict(feat)
                else:
                    feat = model(images)
                    output = classifier(feat)
            else:
                if config.dataset == 'imagenet':
                    feat = model(images)
                    if config.fixed_classifier:
                        output = classifier.module.predict(feat)
                else:
                    feat = model(images)
                    if config.fixed_classifier:
                        output = classifier.predict(feat)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)
        acc_classes = correct / class_num
        if config.dataset.endswith('LT'):
            head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100
            med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
            tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
            logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))
        else:
            logger.info(
            '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% '.format(top1=top1, top5=top5))
        cal = calibration(true_class, pred_class, confidence, num_bins=15)


    return top1.avg, cal['expected_calibration_error'] * 100

def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)

def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    config.gpu = gpu

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar100':
            model = getattr(resnet32_balancedC100, config.backbone)(depth=32, output_dims= config.space_dim, multiplier= 1)
    elif config.dataset == 'imagenet200':
            model = getattr(resnet32_balancedIN200, config.backbone)(depth=32, output_dims=config.space_dim, multiplier=1)
    elif config.dataset == 'cifar10LT' or config.dataset == 'cifar100LT' or config.dataset == 'stl10LT' or config.dataset == 'svhnLT':
        model = getattr(resnet_LT, config.backbone)()
    elif config.dataset == 'imagenetLT':
        model = getattr(resnet_IN, config.backbone)()
    elif config.dataset == 'imagenet':
        model = getattr(largenet, 'net')(output_dim= config.space_dim, model_name=config.backbone)

    if config.fixed_classifier:
        print('########   Using a fixed hyperspherical classifier with DL2PA  ##########')
        classifier = getattr(fixed, 'fixed_Classifier')(feat_in=config.space_dim, num_classes=config.num_classes, centroid_path=config.centroid_path, gpu_id=config.gpu)
    else:
        classifier = getattr(learnable, 'Classifier')(feat_in=config.space_dim, num_classes=config.num_classes, gpu_id=config.gpu)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)

    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)

    else:
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

        # define loss function (criterion) and optimizer
    if config.fixed_classifier:
        criterion = nn.CosineSimilarity(eps=1e-9).cuda(config.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(config.gpu)

    # Data loading code
    if config.dataset == 'cifar100':
        trainloader, testloader = dataset.load_cifar100(config.data_path, config.batch_size, {'num_workers': config.workers, 'pin_memory': True})
    elif config.dataset == 'imagenet200':
        trainloader, testloader = dataset.load_imagenet200(config.data_path, config.batch_size, {'num_workers': config.workers, 'pin_memory': True})
    elif config.dataset == 'cifar10LT':
        trainloader, testloader = dataset.CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'cifar100LT':
        trainloader, testloader = dataset.CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'stl10LT':
        trainloader, testloader = dataset.STL10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                                                   batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'svhnLT':
        trainloader, testloader = dataset.SVHN_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                          batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'imagenetLT':
        trainloader, testloader = dataset.ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'imagenet':
        trainloader, testloader = dataset.ImageNet(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)
        scaler = GradScaler()

    if config.distributed:
        train_sampler = dataset.dist_sampler

    if config.backbone != 'swin_tiny_patch4_window7_224':
        optimizer = torch.optim.SGD([{"params": model.parameters()},
                                    {"params": classifier.parameters()}], config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW([{"params": model.parameters()},
                                    {"params": classifier.parameters()}], lr=config.lr,
                                      weight_decay=config.weight_decay)

    lr = config.lr
    best_acc1 = -1.0

    if config.dataset == 'imagenet':
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs - config.warmup_epochs  # , eta_min = 1e-5
        )
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.lr_warmup_decay, total_iters=config.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.warmup_epochs]
        )
        scaler = GradScaler()

    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # adjust learning rate
        if config.dataset == 'cifar100' or config.dataset == 'imagenet200':
            if epoch in [config.drop1, config.drop2]:
                lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        elif config.dataset == 'cifar10LT' or config.dataset == 'cifar100LT' or config.dataset == 'stl10LT' or config.dataset == 'svhnLT' or config.dataset == 'imagenetLT':
            """Sets the learning rate"""
            epoch = epoch + 1
            if epoch <= config.drop1:
                lr = config.lr * epoch / 5
            elif epoch > config.drop3:
                lr = config.lr * 0.01
            elif epoch > config.drop2:
                lr = config.lr * 0.1
            else:
                lr = config.lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # train for one epoch
        if config.dataset != 'imagenet':
            train(trainloader, model, classifier, criterion, optimizer, epoch, config, logger)
        else:
            train(trainloader, model, classifier, criterion, optimizer, epoch, config, logger,scheduler,scaler)

        # evaluate on validation set
        acc1, ece = validate(testloader, model, classifier, config, logger)

        # hungarian
        if config.dataset == 'imagenet':
            classifier.module.update_fixed_center()
        else:
            classifier.update_fixed_center()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logger.info('Best Prec@1: %.3f%% \n' % (best_acc1))

        if config.fixed_classifier:
            if config.dataset == 'imagenet':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'polars': classifier.module.polars,
                    'best_acc1': best_acc1,
                    'its_ece': its_ece,
                }, is_best, model_dir)
            else:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_classifier': classifier.state_dict(),
                'polars': classifier.polars,
                'best_acc1': best_acc1,
                'its_ece': its_ece,
            }, is_best, model_dir)
        else:
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict_model': model.state_dict(),
                            'state_dict_classifier': classifier.state_dict(),
                            'best_acc1': best_acc1,
                            'its_ece': its_ece,
                        }, is_best, model_dir)

if __name__ == '__main__':
    main()