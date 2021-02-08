import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorboardX

from config import parser, args_augment
from utils import Monitor, accuracy

from datasets.MNIST_ADD.mnistadd import MNISTAdd, train_test_splits
from model.logic_net_rf import LogicNet

logger = None
summary_writter = None
args = None


def build_model(args):
    model = LogicNet(args)
    logger.info(model)
    return model


def train(model, optim, trainloader, epoch, monitors):
    model.train()
    logger.info("Training starts")
    for batch_idx, (image, target, meta_data) in enumerate(trainloader):
        # Data
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            # meta_data = [data.cuda() for data in meta_data]
        # Learn
        # exit()
        # img = image[0, 0].unsqueeze(0) + 0.5
        # summary_writter.add_image(
        #             "train-img", img.cpu().detach(), epoch * len(trainloader) + batch_idx
        #         )
        optim.zero_grad()
        loss, pred, log = model.compute_loss(image, target, meta_data)
        loss.backward()
        optim.step()
        # Monitor
        log_str = []
        log_str.append(monitors['loss'].update_and_format(loss.item(), summary_writter))
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
    # import pdb; pdb.set_trace()
    average_loss, average_acc = monitors['loss'].reset_and_log(), monitors['acc'].reset_and_log()
    # # import pdb; pdb.set_trace()
    logger.infox("Average Training Loss: {:.6f}, Acc: {:.6f}".format(average_loss, average_acc))


def validate(model, validloader, epoch, monitors):
    model.eval()
    logger.info("Validation starts")
    for batch_idx, (image, target, meta_data) in enumerate(validloader):
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
        with torch.no_grad():
            loss, pred, log = model.compute_loss(image, target, meta_data)
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

    train_split, valid_split = train_test_splits(split_train_test=False)
    # import pdb; pdb.set_trace()
    train_dataset = MNISTAdd(args.path, args.dataset_size, train_split, 'rd', train=True)
    valid_dataset = MNISTAdd(args.path, args.dataset_size, valid_split, 'rd', train=False)

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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Monitors
    train_monitors = {
        'acc': Monitor("training/acc"),
        'loss': Monitor("training/loss"),
    }
    valid_monitors = {
        'acc': Monitor("val/acc"),
        'loss': Monitor('val/loss'),
    }

    for epoch in range(0, args.epochs):
        logger.info("=" * 80)
        train(model, optimizer, trainloader, epoch, train_monitors)
        avg_loss, avg_acc = validate(model, validloader, epoch, valid_monitors)
        model.save(args.save, epoch, avg_acc, avg_loss)


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
