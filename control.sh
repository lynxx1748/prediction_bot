#!/bin/bash

# Path to virtual environment
VENV_PATH="$(pwd)/venv"

# Ensure virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found! Please run setup.sh first."
    exit 1
fi

# Function to activate virtual environment
activate_venv() {
    source "$VENV_PATH/bin/activate"
}

case "$1" in
    vpn)
        case "$2" in
            connect)
                ./vpn.sh connect
                ;;
            disconnect)
                ./vpn.sh disconnect
                ;;
            status)
                ./vpn.sh status
                ;;
            *)
                echo "Usage: $0 vpn {connect|disconnect|status}"
                exit 1
                ;;
        esac
        ;;
    start)
        # Check VPN first
        if ! ./vpn.sh status | grep -q "Connected"; then
            echo "⚠️ VPN not connected! Connecting..."
            ./vpn.sh connect
        fi
        echo "Starting UglyBot..."
        activate_venv
        sudo systemctl start uglybot
        echo "Starting Web Interface..."
        sudo systemctl start uglybot-web
        ;;
    # ... rest of the control script remains the same ...
esac 