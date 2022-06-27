import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Union

from latch import large_task, medium_task, small_task, workflow
from latch.types import LatchDir, LatchFile
from typing import Optional


@small_task
def unirep_tutorial(
    use_full_1900_dim_model: bool = False,
    ) -> LatchDir:
    print("INIZIO")
    local_dir = "/root/local_dir/"
    remote_dir = "latch:///test_unirep/"
    _cmd = [
        "python3",
        "tutorial.py",
        str(use_full_1900_dim_model)
        ]
    print("DURANTE")

    subprocess.run(_cmd)
    print("FINE")
    return LatchDir(local_dir, remote_dir)

@workflow
def unirep(
    use_full_1900_dim_model: bool = False,
    ) -> LatchDir:

    """
    Args:
        use_full_1900_dim_model:
            Use 1900 model rather than base 64 one
            __metadata__:
            display_name: Use 1900 model rather than base 64 one
    """
    return(unirep_tutorial(use_full_1900_dim_model=use_full_1900_dim_model))
