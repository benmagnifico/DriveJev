"""DriveJev: a frozen Qwen-Drive-1.0 VLM + a small decision head that chooses among offered
semantic driving behaviours; a local executor turns the choice into steering and speed."""
__version__ = "1.0.0"

from .compiler import COMPILER_VERSION, DriveCompiler  # noqa: F401
from .policy import DrivingPolicy  # noqa: F401
