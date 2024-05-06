# Monkey patching is a technique to add, modify,
# or suppress the default behavior of a piece of code at runtime
# without changing its original source code.

# VSCode use repr to show variable info.

BACKUP = {}


def replace_torch_tensor_repr():
    import torch
    original_repr = torch.Tensor.__repr__
    BACKUP["torch.Tensor.__repr__"] = original_repr

    def tensor_repr(self, *, tensor_contents=None):
        assert isinstance(self, torch.Tensor)
        s = f"Tensor {list(self.shape)} {self.dtype}"
        s += f" {self.device}"
        s += " requires grad" if self.requires_grad else ""
        s += "\n" + original_repr(self)
        return s

    torch.Tensor.__repr__ = tensor_repr


def restore_torch_tensor_repr():
    import torch
    torch.Tensor.__repr__ = BACKUP["torch.Tensor.__repr__"]


def replace_torch_parameter_repr():
    import torch
    original_repr = torch.nn.Parameter.__repr__
    BACKUP["torch.nn.Parameter.__repr__"] = original_repr

    def parameter_repr(self, *, tensor_contents=None):
        assert isinstance(self, torch.Tensor)
        s = f"Parameter {list(self.shape)} {self.dtype}"
        s += " requires grad" if self.requires_grad else ""
        s += "\n" + original_repr(self)
        return s

    torch.nn.Parameter.__repr__ = parameter_repr


def restore_torch_parameter_repr():
    import torch
    torch.nn.Parameter.__repr__ = BACKUP["torch.nn.Parameter.__repr__"]


replace_torch_tensor_repr()
replace_torch_parameter_repr()

