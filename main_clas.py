import os
import argparse

import torch
from torch.utils.data import DataLoader


from MNIST_RPM.dataset import MNIST_RPM
from model.cnn_classifier import CNN_MLP


parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='cnn_mlp')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default='./MNIST_RPM/')
parser.add_argument('--save', type=str, default='./experiments/checkpoint/')
parser.add_argument('--dataset_size', type=int, default=30000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_alpha', type=float, default=0.0)
parser.add_argument('--meta_beta', type=float, default=0.0)
# Model parameters
parser.add_argument('--input_size', type=int, default=28)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--ans_dim', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=3)
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

train = MNIST_RPM(args.path, args.dataset_size, args.seq_len, train=True)
valid = MNIST_RPM(args.path, args.dataset_size, args.seq_len, train=False)

trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=16)


model = CNN_MLP(args)
    
if args.resume:
    model.load_model(args.save, 0)
    print('Loaded model')
if args.cuda:
    model = model.cuda()


def train(epoch):
    model.train()

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image_mat, answer_mat, target, attr_mat) in enumerate(trainloader):
        counter += 1
        image_mat = image_mat.float() / 255
        answer_mat = answer_mat.float() / 255
        if args.cuda:
            image_mat = image_mat.cuda()
            answer_mat = answer_mat.cuda()
            target = target.cuda()
            attr_mat = attr_mat.cuda()
        loss, acc1, acc2, acc3 = model.train_(image_mat, answer_mat, target, attr_mat)
        print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}, {:.4f}, {:.4f}.'.format(
            epoch, batch_idx, loss, acc1, acc2, acc3))
        loss_all += loss
        acc_all += (acc1 + acc2 + acc3) / 3
    if counter > 0:
        print("Avg Training Loss: {:.6f}, Acc: {:.6f}".format(
            loss_all / float(counter), acc_all / float(counter)))


def validate(epoch):
    model.eval()

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image_mat, answer_mat, target, attr_mat) in enumerate(validloader):
        counter += 1
        image_mat = image_mat.float() / 255
        answer_mat = answer_mat.float() / 255
        if args.cuda:
            image_mat = image_mat.cuda()
            answer_mat = answer_mat.cuda()
            target = target.cuda()
            attr_mat = attr_mat.cuda()
        loss, acc1, acc2, acc3 = model.train_(image_mat, answer_mat, target, attr_mat)
        print('Validate: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}, {:.4f}, {:.4f}.'.format(
            epoch, batch_idx, loss, acc1, acc2, acc3))
        loss_all += loss
        acc_all += (acc1 + acc2 + acc3) / 3
    if counter > 0:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(
            loss_all / float(counter), acc_all / float(counter))
        )
    return loss_all / float(counter), acc_all / float(counter)


def test(epoch):
    model.eval()

    acc_all = 0.0
    counter = 0
    for batch_idx, (image_mat, answer_mat, target, attr_mat) in enumerate(testloader):
        counter += 1
        if args.cuda:
            image_mat = image_mat.cuda()
            answer_mat = answer_mat.cuda()
            target = target.cuda()

        loss, acc = model.test_(image_mat, answer_mat, target)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))
        acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all / float(counter)


def main():
    for epoch in range(0, args.epochs):
        train(epoch)
        avg_loss, avg_acc = validate(epoch)
        model.save_model(args.save, epoch, avg_acc, avg_loss)


if __name__ == '__main__':
    main()
