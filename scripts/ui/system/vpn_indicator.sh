#!/bin/bash
# VPN status indicator and control script

# Function to check VPN status
check_vpn_status() {
    STATUS=$(protonvpn-cli status | grep Status | cut -d':' -f2 | tr -d ' ')
    echo "$STATUS"
}

# Function to show notification
show_notification() {
    if [ "$1" == "Connected" ]; then
        notify-send -i network-vpn "Trading VPN" "Connected to Secure Network" -t 3000
    else
        notify-send -i network-vpn-offline "Trading VPN" "Not Connected - Trading May Be Restricted" -t 3000
    fi
}

# Function to display status icon
show_status_icon() {
    if [ "$1" == "Connected" ]; then
        yad --notification --image=network-vpn --text="VPN: Connected" \
            --command="zenity --list --title='VPN Control' --text='Select VPN Action' \
            --column='Action' 'Disconnect' 'Status' | xargs -I {} $HOME/ugly-bot/vpn.sh {}"
    else
        yad --notification --image=network-vpn-offline --text="VPN: Disconnected" \
            --command="zenity --list --title='VPN Control' --text='Select VPN Action' \
            --column='Action' 'Connect' 'Status' | xargs -I {} $HOME/ugly-bot/vpn.sh {}"
    fi
}

# Main loop
while true; do
    VPN_STATUS=$(check_vpn_status)
    show_notification "$VPN_STATUS"
    show_status_icon "$VPN_STATUS"
    
    # Check every 30 seconds
    sleep 30
done 