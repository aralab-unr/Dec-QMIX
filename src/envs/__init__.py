from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .wildfire import WildfireEnvironment

# try:
#     gfootball = True
#     from .gfootball import GoogleFootballEnv
#     print("Opted for Football.")
# except Exception as e:
#     gfootball = False
#     print("Opted for Football, but exception araised.")
#     print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["wildfire"] = partial(env_fn, env=WildfireEnvironment)

# if gfootball:
#     REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
