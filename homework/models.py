import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=6):
        super(LinearClassifier, self).__init__()

        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=6):
        super(MLPClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 64)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 6)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
