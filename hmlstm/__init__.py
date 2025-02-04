# TODO change implementations to torchscript (performance)

from .state import HMLSTMState, HMLSTMStatesList

# define which version is active
from .utils import Round1 as Round, HardSigm1 as HardSigm, SlopeScheduler

# define which version is active
from .cell import HMLSTMCell4 as LayerNormHMLSTMCell, HMLSTMCell2 as HMLSTMCell

from .model import HMLSTM
from .output import HMLSTMOutput
from .network import HMLSTMNetwork


__version__     = '0.2'
__author__      = "Clemens Kriechbaumer"
__email__       = "clemens.kr@gmail.com"
__date__        = "Feb 2020"
__copyright__   = "Copyright 2020"

__info__ = "implementation based on - " \
           "J. Chung, S. Ahn and Y. Bengio, 'Hierarchical Multiscale Recurrent Neural Networks,' arXiv:1609.01704, 2016. "


