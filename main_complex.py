import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorboardX

from config import parser, args_augment
from utils import Monitor, accuracy

from datasets.MNIST_RULE.dataset import MNISTRPM
from model.DLN_complex import DLN

logger = None
summary_writter = None
args = None


def build_model(args):
    model = DLN(args)
    return model


pretrain_batch = 100


def train(model, optim_theta, optim_phi, trainloader, epoch, monitors):
    model.train()
    logger.info("Training starts")

    model.resample_structure()

    for batch_idx, (image, meta_data, logics) in enumerate(trainloader):
        # Data
        if args.cuda:
            image = image.cuda()
            meta_data = {key: val.cuda() for key, val in meta_data.items()}
            logics = {key: val.cuda() for key, val in logics.items()}
        total_loss = 0
        global pretrain_batch
        if pretrain_batch > 0:
            optim_theta.zero_grad()
            loss, log = model.compute_pretrain_loss(image, meta_data, logics)
            total_loss = loss
            pretrain_batch = pretrain_batch - 1
            loss.backward()
            optim_theta.step()
        else:
            optim_phi.zero_grad()
            optim_theta.zero_grad()
            loss_phi, loss_theta, log = model.compute_overall_loss(image, meta_data, logics)
            total_loss = loss_phi + loss_theta
            if loss_phi != 0:
                loss_phi.backward()
                optim_phi.step()
            if loss_theta != 0:
                loss_theta.backward()
                optim_theta.step()

        # Monitor
        log_str = []
        log_str.append(monitors['loss'].update_and_format(total_loss.item(), summary_writter))
        for key, val in log.items():
            if key in monitors:
                log_str.append(monitors[key].update_and_format(val, summary_writter))
        # log distribution information
        # Logging
        if batch_idx % args.log_interval == 0:

            logger.info('Train: Epoch:{}, Batch:{:04d}/{:04d}, {}'.format(
                epoch, batch_idx, len(trainloader), ', '.join(log_str)
            ))
        # if batch_idx % 100 == 99 and train_structure:
        #     model.anneal(summary_writter)
    logger.info("*" * 50)
    for key, val in model.deeplogics.items():
        logger.info("Top K rule system of logic[{}]:".format(key))
        val.vis_sequence()
    logger.info("*" * 50)
    average_loss = monitors['loss'].reset_and_log()
    average_acc = sum([monitors[key].reset_and_log() for key in log if key.startswith("acc")]) / 3
    logger.infox("Average Training Loss: {:.6f}, Acc: {:.6f}".format(average_loss, average_acc))


def validate(model, validloader, epoch, monitors):
    model.eval()
    model.apply_best_model()
    logger.info("Validation starts")
    for batch_idx, (image, meta_data, logics) in enumerate(validloader):
        if args.cuda:
            image = image.cuda()
            meta_data = {key: val.cuda() for key, val in meta_data.items()}
            logics = {key: val.cuda() for key, val in logics.items()}
  
        with torch.no_grad():
            loss, pred, log = model.compute_perception_loss(image, meta_data, logics)
        # Monitor
        log_str = []
        log_str.append(monitors['loss'].update_and_format(loss.item(), summary_writter))
        for key, val in log.items():
            if key in monitors:
                log_str.append(monitors[key].update_and_format(val, summary_writter))
        # Logging
        if batch_idx % args.log_interval == 0:
            logger.info('Validate: Epoch:{}, Batch:{:04d}/{:04d}, {}'.format(
                epoch, batch_idx, len(validloader), ', '.join(log_str)
            ))
    average_loss, average_acc = monitors['loss'].reset_and_log(), monitors['acc'].reset_and_log()
    logger.infox("Average Validation Loss: {:.6f}, Acc: {:.6f}".format(average_loss, average_acc))
    return average_loss, average_acc


def main(args):
    logger.info("Starting...")
    logger.info("Manual Seed: {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # self, root_dir, dataset_size, seq_len, train=True, image_num=10000

    # import pdb; pdb.set_trace()
    train_dataset = MNISTRPM("./datasets", args.dataset_size, 3, train=True)
    valid_dataset = MNISTRPM("./datasets", args.dataset_size, 3, train=False)

    trainloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Models
    model = build_model(args)
    if args.cuda:
        model = model.cuda()
    logger.info("*" * 20 + "Model:" + "*" * 20)
    print(model)
    logger.info("*" * 20 + "Model:" + "*" * 20)
    # model.resample_structure()  # call this function after the .cuda() command

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim_theta, optim_phi = model.obtain_optimizer()
    # Monitors
    train_monitors = {
        'acc_NUMBER': Monitor("training/acc/number"),
        'acc_COLOR': Monitor("training/acc/color"),
        'acc_BCOLOR': Monitor("training/acc/bcolor"),
        'lacc_NUMBER': Monitor("training/lacc/number"),
        'lacc_COLOR': Monitor("training/lacc/color"),
        'lacc_BCOLOR': Monitor("training/lacc/bcolor"),
        'loss': Monitor("training/loss"),
        'loss_NUMBER': Monitor("training/losses/NUMBER"),
        'loss_COLOR': Monitor("training/losses/COLOR"),
        'loss_BCOLOR': Monitor("training/losses/BCOLOR"),
        'reward_L1': Monitor("training/reward_l1"),
        'reward_L2': Monitor("training/reward_l2"),
        'reward_L3': Monitor("training/reward_l3"),
    }
    valid_monitors = {
        'acc_NUMBER': Monitor("val/acc/number"),
        'acc_COLOR': Monitor("val/acc/color"),
        'acc_BCOLOR': Monitor("val/acc/bcolor"),
        'lacc_NUMBER': Monitor("val/lacc/number"),
        'lacc_COLOR': Monitor("val/lacc/color"),
        'lacc_BCOLOR': Monitor("val/lacc/bcolor"),
        'loss': Monitor("val/loss"),
        'loss_NUMBER': Monitor("val/losses/NUMBER"),
        'loss_COLOR': Monitor("val/losses/COLOR"),
        'loss_BCOLOR': Monitor("val/losses/BCOLOR"),
        'reward_L1': Monitor("val/reward_l1"),
        'reward_L2': Monitor("val/reward_l2"),
        'reward_L3': Monitor("val/reward_l3"),
    }

    global pretrain_batch
    for epoch in range(-10, 20):
        train(model, optim_theta, optim_phi, trainloader, epoch, train_monitors)
        # avg_loss, avg_acc = validate(model, validloader, epoch, valid_monitors)

    # for epoch in range(0, args.epochs):
    #     logger.info("=" * 80)
    #     train(model, optim_theta, optim_phi, trainloader, epoch, train_monitors, flag=False)
    #     avg_loss, avg_acc = validate(model, validloader, epoch, valid_monitors)
    #     model.save(args.save, epoch, avg_acc, avg_loss)
 

if __name__ == '__main__':
    parser = parser()
    args = args_augment(parser.parse_args())
    logger = logging.getLogger('model')
    summary_writter = tensorboardX.SummaryWriter(args.logdir)
    logger.info("*" * 80)
    logger.info(vars(args))
    # logger.info(json.dumps(vars(args.model), sort_keys=True, indent=2, separators=(',', ': ')))
    logger.info("*" * 80)
    main(args)
