import torch

class MSE:
    """均方误差"""
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Smooth(torch.nn.Module):
    def __init__(self):
        super(Smooth, self).__init__()

    def forward(self, y_field):
        """平滑形变场"""
        if len(y_field.shape) == 4:
            dy = y_field[:, :, 1:, :] - y_field[:, :, :-1, :]
            dx = y_field[:, :, :, 1:] - y_field[:, :, :, :-1]
            dx = torch.mul(dx, dx)
            dy = torch.mul(dy, dy)
            return (torch.mean(dx) + torch.mean(dy)) / 2.0
        else:
            dy = y_field[:, :, :, 1:, :] - y_field[:, :, :, :-1, :]
            dx = y_field[:, :, :, :, 1:] - y_field[:, :, :, :, :-1]
            dz = y_field[:, :, 1:, :, :] - y_field[:, :, :-1, :, :]
            dx = torch.mul(dx, dx)
            dy = torch.mul(dy, dy)
            dz = torch.mul(dz, dz)
            return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

    def loss(self, y_field):
        return self.forward(y_field)


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )

stablestd = StableStd.apply

class NCC(torch.nn.Module):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask

    @staticmethod
    def ncc(x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = stablestd(x1) * stablestd(x2)
        return cc / (std + e)

    @staticmethod
    def ncc_mask(x1, x2, mask, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        x1 = torch.masked_select(x1, mask)
        x2 = torch.masked_select(x2, mask)
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = stablestd(x1) * stablestd(x2)
        return cc / (std + e)

    def loss(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return 1 - self.ncc(fixed, warped)

    def masked_loss(self, fixed: torch.Tensor, warped: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return 1 - self.ncc_mask(fixed, warped, mask)

    @staticmethod
    def metric(fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return NCC.ncc(fixed, warped)

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor, mask=None):
        if self.use_mask and mask is not None:
            return self.masked_loss(fixed, warped, mask)
        return self.loss(fixed, warped)