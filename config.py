import argparse
import os
import random
import logging
import sys

import yaml
import numpy as np
import torch
from termcolor import colored
from functools import partial, partialmethod


# CONST for more logger level
logging.INFOX = logging.INFO + 5
logging.addLevelName(logging.INFOX, 'INFOX')
logging.Logger.infox = partialmethod(logging.Logger.log, logging.INFOX)
logging.infox = partial(logging.log, logging.INFOX)


def set_device(id=-1):
    if id == -1:
        tmp_file_name = 'tmp%s' % (random.random())
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >%s' % (tmp_file_name))
        memory_gpu = [int(x.split()[2]) for x in open(tmp_file_name, 'r').readlines()]
        id = np.argmax(memory_gpu)
        os.system('rm %s' % (tmp_file_name))
    return id


class _dict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def convert(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    data[k] = _dict(v)
                    _dict.convert(data[k])
            return _dict(data)
        else:
            return data


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        # import pdb; pdb.set_trace()
        if record.levelno == logging.INFOX:
            prefix = colored("INFOX", 'blue', attrs=['blink', 'underline'])
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="logic_tree", abbrev_name=""
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)

        fh = logging.StreamHandler(open(filename, 'a'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def parser():
    parser = argparse.ArgumentParser(description='our_model')
    # parser.add_argument('--model', type=str, default='Resnet18_MLP')
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--seed', type=int, default=12345)
    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--load_workers', type=int, default=16)
    # parser.add_argument('--resume', type=bool, default=False)
    # parser.add_argument('--path', type=str, default='~/logical_tree/data/RAVEN-10000/')
    # parser.add_argument('--save', type=str, default='./save/')
    # parser.add_argument('--img_size', type=int, default=223)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--beta1', type=float, default=0.9)
    # parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    # parser.add_argument('--meta_alpha', type=float, default=0.0)
    # parser.add_argument('--meta_beta', type=float, default=0.0)
    # validate
    parser.add_argument("--validate", help="Validating", default=False, action="store_true")
    # Training parameters
    training = parser.add_argument_group("training", "Training settting")
    training.add_argument('--epochs', type=int, default=200,
        help="Training epochs. In each epoch, the whole model will be trained")
    training.add_argument('--batch_size', type=int, default=32,
        help="Batch Size.")
    training.add_argument("--seed", type=int, default=12345,
        help="Random Seeds for all modules (torch, numpy, random)")
    training.add_argument("--device", type=int, default=-1,
        help="GPU devices. (Considering setting devices automatically)")
    training.add_argument("--lr", type=float, default=0.1,
        help="Learning Rate")
    training.add_argument("--num_workers", type=int, default=16,
        help="number of workers used to prepare data")
    training.add_argument("--log_interval", type=int, default=10,
        help="logging intervals")

    # Data Parameters
    data = parser.add_argument_group("data", "Data settting")
    data.add_argument("--dataset_size", type=int, default=5000)
    data.add_argument("--path", type=str, default="./datasets",
        help="Raw data path. You should download the data")
    data.add_argument("--prefix", type=str, default='*',
        help="prefix for the model")
    data.add_argument("--save", type=str, default="./save/",
        help="Save dir for models and results")
    data.add_argument("--resume", type=bool, default=False,
        help="Whether to use the resumed model or not.")
    data.add_argument("--alias", type=str, default="test",
        help="Alias for the model, which is important for model saving")
    
    # Model Parameters
    model = parser.add_argument_group('model', "model structures")
    model.add_argument("--img_size", type=list, default=[80, 80])
    model.add_argument("--pretrain", default=False, action="store_true",
        help="Whether to train the whole model or to pretrain the encoder")
    model.add_argument("--model", type=str, default='./config/MNIST_ADD_BASIC.yaml',
        help="model yaml file")
    model.add_argument("--encoder_weight", type=str, default='',
        help="path to the pretrained encoder")
    model.add_argument("--weight", type=str, default='',
        help="path to the trained model")
    return parser


def args_augment(args):

    # CUDA
    args.cuda = torch.cuda.is_available()
    if args.cuda and args.device == -1:
        torch.cuda.set_device("cuda:%d" % set_device())
    elif not args.cuda:
        assert False, "This code requires cuda environment!"
    else:
        torch.cuda.set_device("cuda:%d" % args.device)
    
    # SAVE
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    args.alias = "{}-{}-{:d}".format(args.alias, args.model.split('/')[-1], args.batch_size)
    args.logdir = os.path.join(args.save, args.alias)
    # import pdb; pdb.set_trace()
    if not args.validate and os.path.exists(args.logdir) and not args.alias.startswith("test"):
        print("Alias in USE")
        exit()
    elif args.alias.startswith("test"):
        args.logdir = "{}-{:03d}".format(args.logdir, random.randint(0, 100))
        if os.path.exists(args.logdir):
            os.system('rm %s -rf' % (args.logdir))
    if not args.validate:
        os.mkdir(args.logdir)
    logger = setup_logger(args.logdir, name='model')
    logger.info("Alias: {}".format(args.alias))
    
    # validation and model:
    if args.validate and (args.weight == ""):
        assert False, "Weight is required for validation"
    # YAML
    with open(args.model, 'r', encoding='utf8') as F:
        args.model = _dict.convert(yaml.load(F.read(), Loader=yaml.FullLoader))
        args.img_size = [int(x) for x in args.model.encoder.input_size]
    return args
