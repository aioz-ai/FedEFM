import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
#from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from cal_metric import jaccard, calculate_miou
from evaluate import evaluate


def trainer_synapse(device, args, model, snapshot_path):
    from datasets.med_dataset import Synapse_dataset, RandomGenerator, MyDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train = MyDataset(images_dir='data/phantom_train/images', mask_dir='data/phantom_train/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_test = MyDataset(images_dir='data/phantom_test/images', mask_dir='data/phantom_test/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = -1 
    metric_list = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    experiment = wandb.init(project='TransU-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=max_epoch, batch_size=batch_size, learning_rate=base_lr,
              save_checkpoint=snapshot_path )
    )




    for epoch_num in iterator:
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            # print (image_batch.shape, label_batch.shape)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            experiment.log({
                    'train loss': loss.item(),
                    'step': iter_num,
                    'epoch': epoch_num
                })
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            #break
            # writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            #logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            
        # performance, jacc, miou = evaluate(model, testloader, device)
        # print(performance, jacc, miou)

        list_pred = [] 
        list_label =[] 
        for test_sample in tqdm(testloader): 
            image, label = test_sample["image"], test_sample["label"]#, sampled_batch['case_name'][0]
            #metric_i,intersection_sample, union_sample  
            metric_i,pred,lab= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None,case=None, z_spacing=1, device = device)
            metric_list += metric_i
            #break
            list_pred.append(pred)
            list_label.append(lab) 

        list_pred = torch.stack(list_pred, dim=0)
        list_label = torch.stack(list_label, dim = 0)

        jacc = jaccard(list_pred, list_label)
        miou = calculate_miou(list_pred, list_label)
        performance = metric_list / len(db_test)
        #performance = np.mean(metric_list, axis=0)[0]
        if miou>best_performance: 
            best_performance=miou
            save_mode_path = os.path.join(snapshot_path, 'best_miou.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("Save best model with the best miou score = {} ".format(performance))
        
        logging.info("Performance at epoch iter {}: {} - {} - {}".format(epoch_num, performance, miou, jacc))
        print(jacc,miou, performance)
        experiment.log({
                    'learning rate': base_lr,
                    "miou": miou,
                    'validation Dice': performance,
                    "jaccard": jacc, 
                    
                    'step': iter_num,
                    'epoch': epoch_num,
                    #**histograms
                })
        save_interval = 20  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    #writer.close()
    return "Training Finished!"