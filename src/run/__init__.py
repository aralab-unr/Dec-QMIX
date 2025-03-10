from .run import run as default_run
from .run_gs import run as default_run_gs
from .run_gs2 import run as default_run_gs2
from .run_gs3 import run as default_run_gs3
from .on_off_run import run as on_off_run
from .dop_run import run as dop_run
from .per_run import run as per_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["default_gs"] = default_run_gs
REGISTRY["default_gs2"] = default_run_gs2
REGISTRY["default_gs3"] = default_run_gs3
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run