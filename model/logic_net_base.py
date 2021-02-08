import torch
import torch.nn as nn

from .networks import CNN, MLP, DeepLogic

from .basic_model import BasicModel

r"""
In this function, we define the basic

"""


class LogicNet(BasicModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = CNN(args.encoder)
        # import pdb; pdb.set_trace()
        self.symbolizers = torch.nn.ModuleList(
            [MLP(args.encoder.output_size, val, name=key)
             for key, val in args.symbolizer.items()]
        )
        self.deeplogic = torch.nn.ModuleList(
            [DeepLogic(val, name=key) for key, val in args.deeplogic.items()]
        )
    
    def compute_loss(self, image, target, meta_data=None):
        pred = self(image)
        loss = None
        return loss

    def logits_to_symbols(self, logits):
        r"""
        logits: B * K * N
        """
        prob = nn.functional.softmax(logits, dim=2)
        y_max, ind = prob.max(dim=2)  # M
        return ind, y_max  # symbol and the prob

    def forward(self, inputs, aux=None):
        """
        inputs: B * K * C * H * W (image shape)
        """
        B, K, C, H, W = inputs.shape
        cnn_feature = self.encoder(inputs.view(B * K, C, H, W)).view(B, K, -1)
        logits = [symbolizer(cnn_feature) for symbolizer in self.symbolizers]
        symbols = [self.logits_to_symbols(logit) for logit in logits]
        results = 

        import pdb; pdb.set_trace()
        # A sampler here sample symbols from logits.
        # A symbol is represented with a symbol and a prob

        results = self.deeplogic(labels)

        return results
