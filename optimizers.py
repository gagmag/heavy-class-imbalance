
import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(p.grad.data.sign()).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(p.grad.data.sign())

                p.data -= group['lr'] * buf

        return loss


class GNSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(GNSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step (parameter update)."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Normalize the gradients along each row
                norm = d_p.norm(p=2, dim=1, keepdim=True)
                d_p_normalized = d_p / norm

                # Check for the existence of the 'momentum_buffer' in state
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p_normalized).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p_normalized, alpha=1)

                # Update the parameters
                p.data.add_(buf, alpha=-group['lr'])

        return loss



class DampedNewton(Optimizer):
    """ Damped Newton method optimizer with Hessian computation. """
    def __init__(self, params, lr=1.0, damping_factor=0.1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if damping_factor <= 0.0:
            raise ValueError("Invalid damping factor: {}".format(damping_factor))
        
        defaults = dict(lr=lr, damping_factor=damping_factor)
        super(DampedNewton, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('DampedNewton does not support sparse gradients')

                # Compute the Hessian
                hessian = self.compute_hessian(p, loss)

                # Add damping to the Hessian
                hessian += group['damping_factor'] * torch.eye(hessian.size(0), device=p.device)

                # Invert the Hessian
                hessian_inv = torch.inverse(hessian)

                # Update parameters
                p.data -= group['lr'] * torch.mv(hessian_inv, grad.view(-1)).view_as(p)

        return loss

    def compute_hessian(self, param, loss):
        """Compute the Hessian of the loss function with respect to the parameters."""
        param_grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]

        print(param_grad.shape)
        print("Here")
        hessian = []
        for grad_elem in param_grad.view(-1):
            # Compute second derivative (Hessian components)
            grad_grad = torch.autograd.grad(grad_elem, param, retain_graph=True)[0]
            hessian.append(grad_grad.view(-1))
        hessian = torch.stack(hessian).reshape(param.size(0), param.size(0))
        return hessian

