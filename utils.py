class Monitor:
    def __init__(self, name):
        self.total_counter = 0
        self.counter = 0
        self.val = 0
        self.name = name

    def update_and_format(self, val, writer=None):
        self.total_counter += 1
        self.counter += 1
        self.val += val
        if writer is not None:
            # import pdb; pdb.set_trace()
            writer.add_scalar(self.name, val, self.total_counter)
        return "{}: {:.04f}".format(self.name.split('/')[-1], val)

    def reset_and_log(self, ):
        average_val = self.val / self.counter
        self.counter = 0
        self.val = 0
        return average_val
        # TODO: Log into files


def accuracy(pred, target):
    correct = (pred == target).sum().float().item()
    return correct * 100.0 / target.size(0)
