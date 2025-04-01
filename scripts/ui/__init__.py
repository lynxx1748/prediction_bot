"""
User interface components for the trading bot.
Contains web interface, desktop GUI, and system integration components.
"""

__all__ = ["start_web_interface", "start_control_panel"]

from .desktop import start_control_panel
from .web import start_web_interface
