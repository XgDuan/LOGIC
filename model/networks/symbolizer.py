r"""
A symbolizer helper to organize and trace the loss and reward for symbols.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.distributions as distributions


class SymbolizerFunction(Function):
    
    @staticmethod
    def forward(ctx, logits, sample_path):
        r"""
        probs: the output of a Linear layer, whose shape should be B * C1 * C2 * ... * N
            we would sampler actions from the last dimension
            and the log probs will be saved for backward gradient computation
        """
        # TODO: using the default distributions.Categorical (considering the gumbel-max trick later)
        sampler = distributions.Categorical(logits=logits)
        # whether to sample the symbhols or not
        if sample_path:
            symbols = sampler.sample()
        else:
            _, symbols = logits.max(dim=-1)
        
        log_probs = sampler.log_prob(symbols)
        ctx.save_for_backward(logits, symbols, log_probs)
        symbols = symbols.float()
        symbols.requires_grad = True
        return symbols
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""
        Here we consider grad_output assome kind of reward with the same shape of symbols
        """
        logits, symbols, log_probs = ctx.saved_tensors
        da_dl = torch.softmax(logits, dim=-1)
        # import pdb; pdb.set_trace()
        symbols = symbols.unsqueeze(dim=-1)
        do_da = grad_output.mean(dim=0).unsqueeze(dim=-1)
        da_dl.scatter_add_(dim=-1, index=symbols, src=-torch.ones_like(symbols).float())
        do_dl = do_da * da_dl
        return do_dl, None


class Symbolizer(nn.Module):
    def __init__(self, sample_path):
        super().__init__()
        self.sample_path = sample_path

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, 'RL symbolizer')

    def forward(self, logits):
        # We only sample the path when training the model and the flag is set
        return SymbolizerFunction.apply(logits, self.sample_path and self.training)
