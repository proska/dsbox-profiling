from .data_profile import Profiler

__all__ = ['Profiler']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore