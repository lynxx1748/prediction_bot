#!/bin/bash
# Welcome script for desktop environment
# Runs various components on startup

# Display system information
xfce4-terminal -e "neofetch --config ~/.config/neofetch/config.conf" &

# Launch application launcher
rofi -show drun &

# Start compositor
compton &

# Launch trading bot web interface
python -m scripts.ui.web &

# Notify user that the system is ready
notify-send -i system-software-update "UglyBot System" "Trading environment is ready" -t 5000 