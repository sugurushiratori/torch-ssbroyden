"""torch-ssbroyden: self-scaled Broyden quasi-Newton optimizers for PyTorch.

``torch.optim.Optimizer`` subclasses with the interface of ``torch.optim.LBFGS``::

    from torch_ssbroyden import LSSBroyden, SSBroyden
    from torch_ssbroyden import SegmentTrainer                   # mini-batch / adaptive points
    from torch_ssbroyden.sampling import Uniform, RAD, RARD      # point samplers for the segments
"""
from . import sampling
from .sampling import RAD, RARD, Uniform
from .segments import SegmentTrainer
from .ssbroyden import DenseSSBroyden, LSSBroyden, SSBroyden, ss_params

__all__ = ["LSSBroyden", "SSBroyden", "DenseSSBroyden", "SegmentTrainer", "ss_params",
           "sampling", "Uniform", "RAD", "RARD"]
__version__ = "0.4.1"
