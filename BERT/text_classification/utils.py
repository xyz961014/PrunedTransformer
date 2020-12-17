import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class WeightedLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, num_heads: int, head_size: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, num_heads * head_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 4
        output = torch.einsum("blnd,ond->blno", input, self.weight.reshape(self.out_features, self.num_heads, self.head_size))
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret


    def extra_repr(self) -> str:
        return 'num_heads={}, head_size={}, out_features={}, bias={}'.format(
            self.num_heads, self.head_size, self.out_features, self.bias is not None
        )



def add_attr_from_dict(obj, dic):
    for key, value in dic.items():
        setattr(obj, key, value)
    return obj
