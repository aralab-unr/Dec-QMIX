REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .n_controller_gs2 import NMAC as NMAC2
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["n_mac_gs2"] = NMAC2
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC