#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Ugly Bot v1.0 Setup...${NC}"

# Create required directories
echo -e "\n${YELLOW}📁 Creating directories...${NC}"
mkdir -p logs data configuration scripts models UI templates/images
echo -e "${GREEN}✓ Directories created${NC}"

# Add logo to templates/images directory
cp logo.png templates/images/

# After creating directories, add virtual environment setup
echo -e "\n${YELLOW}🐍 Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Check Python installation
echo -e "\n${YELLOW}🐍 Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found! Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
else
    echo -e "${GREEN}✓ Python3 is installed${NC}"
fi

# Install required packages
echo -e "\n${YELLOW}📦 Installing required packages...${NC}"
pip3 install web3 \
    pandas \
    numpy \
    requests \
    scikit-learn \
    joblib \
    flask \
    psutil \
    python-binance \
    python-dotenv \
    websocket-client \
    technical-analysis \
    plotly \
    dash \
    ccxt

# Also add system dependencies
# Install system dependencies
apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    ufw \
    nginx \
    openvpn \
    dialog \
    python3-proton-client \
    network-manager \
    network-manager-openvpn

# Install ProtonVPN CLI
echo -e "\n${YELLOW}📡 Setting up ProtonVPN...${NC}"
wget https://repo.protonvpn.com/debian/dists/stable/main/binary-all/protonvpn-stable-release_1.0.3_all.deb
dpkg -i protonvpn-stable-release_1.0.3_all.deb
apt-get update
apt-get install -y protonvpn-cli
rm protonvpn-stable-release_1.0.3_all.deb

# Check if config files exist
echo -e "\n${YELLOW}📝 Checking configuration files...${NC}"
if [ ! -f "configuration/config.json" ]; then
    echo -e "${RED}⚠️  config.json not found! Please add your configuration file to configuration/config.json${NC}"
fi

if [ ! -f "configuration/abi.json" ]; then
    echo -e "${RED}⚠️  abi.json not found! Please add your ABI file to configuration/abi.json${NC}"
fi

# Create service file for autostart
echo -e "\n${YELLOW}🔄 Creating service file for auto-start...${NC}"
echo "[Unit]
Description=Ugly Bot v1 Prediction Bot
After=network.target

[Service]
ExecStartPre=/usr/bin/python3 -c "from scripts.web_interface import app; from threading import Thread; Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()"
ExecStart=$(pwd)/venv/bin/python $(pwd)/pp.py
WorkingDirectory=$(pwd)
StandardOutput=append:$(pwd)/logs/bot.log
StandardError=append:$(pwd)/logs/error.log
Restart=always
User=$USER

[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/uglybot.service > /dev/null

# Set permissions
sudo chmod 644 /etc/systemd/system/uglybot.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable uglybot.service

# Add web interface service
echo -e "\n${YELLOW}🌐 Creating web interface service...${NC}"
echo "[Unit]
Description=Ugly Bot Web Interface
After=network.target

[Service]
ExecStart=$(pwd)/venv/bin/python $(pwd)/scripts/web_interface.py
WorkingDirectory=$(pwd)
StandardOutput=append:$(pwd)/logs/web.log
StandardError=append:$(pwd)/logs/web_error.log
Restart=always
User=$USER

[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/uglybot-web.service > /dev/null

# Set permissions for web service
sudo chmod 644 /etc/systemd/system/uglybot-web.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable uglybot-web.service

# Create control script
echo -e "\n${YELLOW}📜 Creating control script...${NC}"
echo '#!/bin/bash

case "$1" in
    start)
        echo "Starting UglyBot..."
        sudo systemctl start uglybot
        echo "Starting Web Interface..."
        sudo systemctl start uglybot-web
        ;;
    stop)
        echo "Stopping UglyBot..."
        sudo systemctl stop uglybot
        echo "Stopping Web Interface..."
        sudo systemctl stop uglybot-web
        ;;
    restart)
        echo "Restarting UglyBot..."
        sudo systemctl restart uglybot
        echo "Restarting Web Interface..."
        sudo systemctl restart uglybot-web
        ;;
    status)
        echo "UglyBot Status:"
        sudo systemctl status uglybot
        echo "Web Interface Status:"
        sudo systemctl status uglybot-web
        ;;
    logs)
        tail -f logs/bot.log
        ;;
    weblogs)
        tail -f logs/web.log
        ;;
    errors)
        tail -f logs/error.log
        ;;
    weberrors)
        tail -f logs/web_error.log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|weblogs|errors|weberrors}"
        exit 1
        ;;
esac' > control.sh

chmod +x control.sh

echo -e "${GREEN}✅ Setup completed!${NC}"
echo -e "\n${YELLOW}To control the bot, use:${NC}"
echo "  ./control.sh start   - Start the bot"
echo "  ./control.sh stop    - Stop the bot"
echo "  ./control.sh restart - Restart the bot"
echo "  ./control.sh status  - Check bot status"
echo "  ./control.sh logs    - View bot logs"
echo "  ./control.sh weblogs  - View web logs"
echo "  ./control.sh errors  - View error logs"
echo "  ./control.sh weberrors - View web error logs"

echo -e "\n${YELLOW}To start the bot automatically:${NC}"
echo "  sudo systemctl start uglybot"

echo -e "\n${RED}Important:${NC}"
echo "1. Make sure to add your configuration files:"
echo "   - configuration/config.json"
echo "   - configuration/abi.json"
echo "2. Review the logs in logs/ directory"
echo "3. The bot will auto-restart if it crashes"

# Create VPN control script
echo -e "\n${YELLOW}📡 Creating VPN control script...${NC}"
echo '#!/bin/bash

case "$1" in
    connect)
        echo "Connecting to ProtonVPN..."
        protonvpn-cli connect
        ;;
    disconnect)
        echo "Disconnecting from ProtonVPN..."
        protonvpn-cli disconnect
        ;;
    status)
        echo "VPN Status:"
        protonvpn-cli status
        ;;
    *)
        echo "Usage: $0 {connect|disconnect|status}"
        exit 1
        ;;
esac' > vpn.sh

chmod +x vpn.sh 