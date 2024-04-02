"""skadi Package"""

from __future__ import annotations

try:
    from .__version__ import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"

from . import safety, allocator, actuator, toolbox, allocator