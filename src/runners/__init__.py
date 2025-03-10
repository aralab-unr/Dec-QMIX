REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_gs import ParallelRunner as ParallelRunner_GS
REGISTRY["parallel_gs"] = ParallelRunner_GS

from .parallel_runner_gs2 import ParallelRunner as ParallelRunner_GS2
REGISTRY["parallel_gs2"] = ParallelRunner_GS2

from .parallel_runner_gs3 import ParallelRunner as ParallelRunner_GS3
REGISTRY["parallel_gs3"] = ParallelRunner_GS3
