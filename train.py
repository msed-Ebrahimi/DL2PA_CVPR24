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
from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter, accuracy
import pprint
import warnings
from backbone.balanced.cifar100 import resnet as resnet32_balanced
from backbone.classifiers import fixed
from backbone.classifiers import learnable
from utils import dataset, calibration,save_checkpoint

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

def train(train_loader, model, classifier, criterion, optimizer, epoch, config, logger):
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
        labels = target.clone()
        # if torch.cuda.is_available():
        #     images = images.cuda(config.gpu, non_blocking=True)
        #     target = target.cuda(config.gpu, non_blocking=True)

        # if config.ETF_classifier:
        #     if config.reg_dot_loss and config.GivenEw:
        #         learned_norm = produce_Ew(target, config.num_classes)
        #         if config.dataset == 'imagenet':
        #             cur_M = learned_norm * classifier.module.polars
        #         else:
        #             cur_M = learned_norm * classifier.polars
        #     else:
        #         if config.dataset == 'imagenet':
        #             cur_M = classifier.module.polars
        #         else:
        #             cur_M = classifier.polars
        #
        # if config.mixup is True:
        #     images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
        #
        #     feat = model(images)
        #     if config.ETF_classifier:
        #         feat = classifier(feat)
        #         output = torch.matmul(feat, cur_M) #+ classifier.module.bias
        #         if config.reg_dot_loss:  ## ETF classifier + DR loss
        #             with torch.no_grad():
        #                 feat_nograd = feat.detach()
        #                 H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
        #             loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
        #             loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
        #             loss = lam * loss_a + (1-lam) * loss_b
        #         else:   ## ETF classifier + CE loss
        #             loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        #
        #     else:       ## learnable classifier + CE loss
        #         output = classifier(feat)
        #         loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        #
        # else:
        #     feat = model(images)
        #     if config.ETF_classifier:
        #         feat = classifier(feat)
        #         output = torch.matmul(feat, cur_M)
        #         if config.reg_dot_loss:   ## ETF classifier + DR loss
        #             with torch.no_grad():
        #                 feat_nograd = feat.detach()
        #                 H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
        #             loss = dot_loss(feat, target, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
        #         else:
        #             loss = criterion(output, target)
        #     else:
        #         output = classifier(feat)
        #         loss = criterion(output, target)

        nlabels = target.clone()
        target = classifier.polars[target]
        if torch.cuda.is_available():
            print('cuda is available')
            images = images.cuda(config.gpu)
            target = target.cuda(config.gpu)
        print('temporary input')
        output = model(images)
        if config.fixed_classifier:
            model.forward_momentum(output.detach(), nlabels.detach())
            loss = (1.0 - criterion(output, target)).pow(2).sum()
            output = model.predict(output, classifier.polars)
        else:
            output = classifier(output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)

def validate(val_loader, model, classifier, criterion, config, logger, dset='test'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    feat_dict = {}
    cnt_dict = {}

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # if config.ETF_classifier: # and config.reg_dot_loss and config.GivenEw:
            #     if config.dataset == 'imagenet':
            #         cur_M = classifier.module.polars
            #     else:
            #         cur_M = classifier.polars

            # compute output

            feat = model(images)
            labels = target.clone()
            target = classifier.polars[target]
            if config.fixed_classifier:
                loss = (1.0 - criterion(output, target)).pow(2).sum()
                output = model.predict(feat,classifier.polars)
            # if config.ETF_classifier:
            #     feat = classifier(feat)
            #     output = torch.matmul(feat, cur_M)
            #     if config.reg_dot_loss:
            #         with torch.no_grad():
            #             feat_nograd = feat.detach()
            #             H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
            #         loss = dot_loss(feat, target, cur_M, classifier, criterion, H_length)
            #     else:
            #         loss = criterion(output, target)
            # else:
            #     output = classifier(feat)
            #     loss = criterion(output, target)
            #
            # if config.stat_mode:
            #     uni_lbl, count = torch.unique(target, return_counts=True)
            #     lbl_num = uni_lbl.size(0)
            #     for kk in range(lbl_num):
            #         sum_feat = torch.sum(feat[torch.where(target==uni_lbl[kk])[0], :], dim=0)
            #         key = uni_lbl[kk].item()
            #         if uni_lbl[kk] in feat_dict.keys():
            #             feat_dict[key] = feat_dict[key]+sum_feat
            #             cnt_dict[key] =  cnt_dict[key]+count[kk]
            #         else:
            #             feat_dict[key] = sum_feat
            #             cnt_dict[key] =  count[kk]



            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(labels, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)
        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
        logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        cal = calibration(true_class, pred_class, confidence, num_bins=15)


        # if config.stat_mode:
        #     ### calculate statistics
        #     #print(feat_dict)
        #     total_sum_feat = sum(feat_dict.values())
        #     total_counts = sum(cnt_dict.values())
        #     global_mean = total_sum_feat / total_counts
        #     class_mean = torch.zeros(config.num_classes, feat.size(1)).cuda()
        #     for i in range(config.num_classes):
        #         if i in feat_dict.keys():
        #             class_mean[i,:] = feat_dict[i] / cnt_dict[i]
        #     class_mean = class_mean - global_mean  ## K, dim
        #     W_mean = classifier.polars.T if config.ETF_classifier else classifier.fc.weight ## K,dim
        #     ## F dist
        #     class_mean_F = class_mean / torch.sqrt(torch.sum(class_mean ** 2))
        #     W_mean_F = W_mean / torch.sqrt(torch.sum(W_mean ** 2))
        #     F_dist =torch.sum((class_mean_F - W_mean_F)**2)
        #     ##
        #     class_mean = class_mean / torch.sqrt(torch.sum(class_mean**2, dim=1, keepdims=True))
        #     W_mean = W_mean / torch.sqrt(torch.sum(W_mean **2, dim=1,keepdims=True))
        #     cos_HH = torch.matmul(class_mean, class_mean.T)
        #     cos_WW = torch.matmul(W_mean, W_mean.T)
        #     cos_HW = torch.matmul(class_mean, W_mean.T)
        #
        #     diag_HW = torch.diag(cos_HW, 0)
        #     diag_avg = torch.mean(diag_HW)
        #     diag_std = torch.std(diag_HW)
        #     ##
        #     up_HH = torch.cat([torch.diag(cos_HH, i) for i in range(1, config.num_classes)])
        #     up_WW = torch.cat([torch.diag(cos_WW, i) for i in range(1, config.num_classes)])
        #     up_HW = torch.cat([torch.diag(cos_HW, i) for i in range(1, config.num_classes)])
        #     up_HH_avg = torch.mean(up_HH)
        #     up_HH_std = torch.std(up_HH)
        #     up_WW_avg = torch.mean(up_WW)
        #     up_WW_std = torch.std(up_WW)
        #     up_HW_avg = torch.mean(up_HW)
        #     up_HW_std = torch.std(up_HW)
        #     ##
        #     print('cos-avg-HH', up_HH_avg)
        #     print('cos-avg-WW', up_WW_avg)
        #     print('cos-avg-HW', up_HW_avg)
        #     print('cos-std-HH', up_HH_std)
        #     print('cos-std-WW', up_WW_std)
        #     print('cos-std-HW', up_HW_std)
        #     ##
        #     print('diag-avg-HW', diag_avg)
        #     print('diag-std-HW', diag_std)
        #     ##
        #     print('||H-M||_F^2', F_dist)
        #     if dset=='train':
        #         cos_avg_HH_train.append(up_HH_avg.item())
        #         cos_avg_WW_train.append(up_WW_avg.item())
        #         cos_avg_HW_train.append(up_HW_avg.item())
        #         cos_std_HH_train.append(up_HH_std.item())
        #         cos_std_WW_train.append(up_WW_std.item())
        #         cos_std_HW_train.append(up_HW_std.item())
        #         HM_F2_train.append(F_dist.item())
        #         diag_avg_HW_train.append(diag_avg.item())
        #         diag_std_HW_train.append(diag_std.item())
        #     else:
        #         cos_avg_HH_val.append(up_HH_avg.item())
        #         cos_avg_WW_val.append(up_WW_avg.item())
        #         cos_avg_HW_val.append(up_HW_avg.item())
        #         cos_std_HH_val.append(up_HH_std.item())
        #         cos_std_WW_val.append(up_WW_std.item())
        #         cos_std_HW_val.append(up_HW_std.item())
        #         HM_F2_val.append(F_dist.item())
        #         diag_avg_HW_val.append(diag_avg.item())
        #         diag_std_HW_val.append(diag_std.item())

    return top1.avg, cal['expected_calibration_error'] * 100

def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        print(f'config.deterministic:{config.deterministic}')
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
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        print(f'config.dist_url:{config.dist_url}')
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        print(f'config.multiprocessing_distributed:{config.multiprocessing_distributed}')
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        print(f'else if ---- config.multiprocessing_distributed:{config.multiprocessing_distributed}')
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)

def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    '''global best_acc1, its_ece'''
    config.gpu = gpu

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        print(f'main worker config.distributed:{config.distributed}')
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
            print(f' config.rank:{config.rank}')
        if config.multiprocessing_distributed:
            print(f'main worker ---> config.multiprocessing_distributed:{config.multiprocessing_distributed}')
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar100' or config.dataset == 'imagenet200':
        model = getattr(resnet32_balanced, config.backbone)(depth=32, output_dims= config.space_dim, multiplier= 1)
        if config.fixed_classifier:
            print('########   Using a fixed hyperspherical classifier instead of Softmax    ##########')
            classifier = getattr(fixed, 'fixed_Classifier')(feat_in=config.space_dim, num_classes=config.num_classes, centroid_path=config.centroid_path)
        else:
            classifier = getattr(learnable, 'Classifier')(feat_in=config.space_dim, num_classes=config.num_classes)

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
        print('using cuda.set_device')
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)

    # Data loading code
    if config.dataset == 'cifar100':
        trainloader, testloader = dataset.load_cifar100(config.data_path, config.batch_size, {'num_workers': config.workers, 'pin_memory': True})
    elif config.dataset == 'imagenet200':
        trainloader, testloader = dataset.load_imagenet200(config.data_path, config.batch_size, {'num_workers': config.workers, 'pin_memory': True})


    # define loss function (criterion) and optimizer
    if config.fixed_classifier:
        criterion = nn.CosineSimilarity(eps=1e-9).cuda(config.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(config.gpu)


    optimizer = torch.optim.SGD([{"params": model.parameters()},
                                {"params": classifier.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    learning_rate = config.lr

    for epoch in range(config.num_epochs):
        # if config.distributed:
        #     train_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, config)
        if config.dataset == 'cifar100' or config.dataset == 'imagenet200':
            if epoch in [config.drop1, config.drop2]:
                learning_rate *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        # train for one epoch

        train(trainloader, model, classifier, criterion, optimizer, epoch, config, logger)

        # evaluate on validation set
        acc1, ece = validate(testloader, model, classifier, criterion, config, logger, 'test')

        # hungarian
        classifier.update_fixed_center()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logger.info('Best Prec@1: %.3f%% \n' % (best_acc1))

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_classifier': classifier.state_dict(),
                'polars': classifier.polars,
                'best_acc1': best_acc1,
                'its_ece': its_ece,
            }, is_best, model_dir)
        # if not config.multiprocessing_distributed or (config.multiprocessing_distributed
        #                                               and config.rank % ngpus_per_node == 0):
        #     if config.dataset != 'imagenet':
        #         if config.ETF_classifier:
        #             save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict_model': model.state_dict(),
        #                 'state_dict_classifier': classifier.state_dict(),
        #                 'cur_M': classifier.ori_M,
        #                 'best_acc1': best_acc1,
        #                 'its_ece': its_ece,
        #             }, is_best, model_dir)
        #         else:
        #             save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict_model': model.state_dict(),
        #                 'state_dict_classifier': classifier.state_dict(),
        #                 'best_acc1': best_acc1,
        #                 'its_ece': its_ece,
        #             }, is_best, model_dir)
        #     else:
        #         if config.ETF_classifier:
        #             save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict_model': model.state_dict(),
        #                 'state_dict_classifier': classifier.state_dict(),
        #                 'cur_M': classifier.module.ori_M,
        #                 'best_acc1': best_acc1,
        #                 'its_ece': its_ece,
        #             }, is_best, model_dir)
        #         else:
        #             save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict_model': model.state_dict(),
        #                 'state_dict_classifier': classifier.state_dict(),
        #                 'best_acc1': best_acc1,
        #                 'its_ece': its_ece,
        #             }, is_best, model_dir)

    # if config.stat_mode:
    #     np.save(model_dir+'/cos_avg_HH_train.npy', np.array(cos_avg_HH_train))
    #     np.save(model_dir+'/cos_avg_WW_train.npy', np.array(cos_avg_WW_train))
    #     np.save(model_dir+'/cos_avg_HW_train.npy', np.array(cos_avg_HW_train))
    #     np.save(model_dir+'/cos_std_HH_train.npy', np.array(cos_std_HH_train))
    #     np.save(model_dir+'/cos_std_WW_train.npy', np.array(cos_std_WW_train))
    #     np.save(model_dir+'/cos_std_HW_train.npy', np.array(cos_std_HW_train))
    #     np.save(model_dir+'/diag_avg_train.npy', np.array(diag_avg_HW_train))
    #     np.save(model_dir+'/diag_std_train.npy', np.array(diag_std_HW_train))
    #     np.save(model_dir+'/HM_F2_train.npy', np.array(HM_F2_train))
    #     ##
    #     np.save(model_dir+'/cos_avg_HH_val.npy', np.array(cos_avg_HH_val))
    #     np.save(model_dir+'/cos_avg_WW_val.npy', np.array(cos_avg_WW_val))
    #     np.save(model_dir+'/cos_avg_HW_val.npy', np.array(cos_avg_HW_val))
    #     np.save(model_dir+'/cos_std_HH_val.npy', np.array(cos_std_HH_val))
    #     np.save(model_dir+'/cos_std_WW_val.npy', np.array(cos_std_WW_val))
    #     np.save(model_dir+'/cos_std_HW_val.npy', np.array(cos_std_HW_val))
    #     np.save(model_dir+'/diag_avg_val.npy', np.array(diag_avg_HW_val))
    #     np.save(model_dir+'/diag_std_val.npy', np.array(diag_std_HW_val))
    #     np.save(model_dir+'/HM_F2_val.npy', np.array(HM_F2_val))

if __name__ == '__main__':
    main()