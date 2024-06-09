import torch
from torch.optim import Optimizer, SGD


class AMP(Optimizer):
    """
    Implements adversarial model perturbation.
    """

    def __init__(self, params, lr, epsilon, inner_lr=1, inner_iter=1, base_optimizer=SGD, **kwargs):
        
        # check for invalid values and raise the errors
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if inner_lr < 0.0:
            raise ValueError(f"Invalid lr: {inner_lr}")
        if inner_iter < 0:
            raise ValueError(f"Invalid iteration: {inner_iter}")
        
        # set default values for the parameters of optimizer
        default_dict = dict(lr=lr, epsilon=epsilon, inner_lr=inner_lr, inner_iter=inner_iter, **kwargs)
        super(AMP, self).__init__(params, default_dict)
        
        # set the param groups as base optimizer param groups
        self.base_optimizer = base_optimizer(self.param_groups, lr=lr, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    
    # step function to optimize 
    def step(self, closure=None):
        if closure is None:
            raise ValueError('Closure not provided for AMP')
        closure = torch.enable_grad()(closure)
        outputs, loss = map(lambda x: x.detach(), closure())
        
        # inner iteration of AMP oprimizer
        for i in range(self.defaults['inner_iter']):
            for value in self.param_groups:
                for p in value['params']:
                    if p.grad is not None:
                        if i == 0:
                            self.state[p]['dev'] = torch.zeros_like(p.grad)
                        dev = self.state[p]['dev'] + value['inner_lr'] * p.grad
                        clip = value['epsilon'] / (dev.norm() + 1e-12)
                        dev = clip * dev if clip < 1 else dev
                        p.sub_(self.state[p]['dev']).add_(dev)
                        self.state[p]['dev'] = dev
            closure()
        
        for value in self.param_groups:
            for p in value['params']:
                if p.grad is not None:
                    p.sub_(self.state[p]['dev'])
        self.base_optimizer.step()
        
        return outputs, loss
