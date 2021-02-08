import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = 'base_model'

    def load_model(self, path, epoch):
        state_dict = torch.load(path + '{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save(self, path, epoch, acc, loss):
        torch.save(
            {'state_dict': self.state_dict(), 'acc': acc, 'loss': loss},
            path + '{}_epoch_{}.pth'.format(self.name, epoch)
        )

    def compute_loss(self, output, target, meta_data):
        pass

    def train_(self, image, candidate, target, meta_data):
        self.optimizer.zero_grad()
        output = self(image, candidate)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, candidate, target, meta_data):
        with torch.no_grad():
            output = self(image, candidate)
        loss = self.compute_loss(output, target)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    # def test_(self, image, target, meta_target, meta_structure, embedding, indicator):
    #     with torch.no_grad():
    #         output = self(image, embedding, indicator)
    #     pred = output[0].data.max(1)[1]
    #     correct = pred.eq(target.data).cpu().sum().numpy()
    #     accuracy = correct * 100.0 / target.size()[0]
    #     return accuracy