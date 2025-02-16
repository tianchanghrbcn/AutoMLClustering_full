#!/bin/bash

# Set working directory
WORK_DIR=$(pwd)

# Step 1: Update apt sources to Tsinghua University mirrors for faster access in China
echo "Updating apt sources to Tsinghua University mirrors..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo bash -c 'cat > /etc/apt/sources.list' << EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
EOF

# Install necessary system libraries
echo "Installing necessary libraries for Baran..."
sudo apt update
sudo apt install -y software-properties-common libatlas-base-dev libblas-dev liblapack-dev gfortran

# Step 2: Install Python 3.9 and set up virtual environment if not already installed
if ! command -v python3.9 &> /dev/null
then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev
fi

# Create virtual environment for Baran and install dependencies
echo "Creating virtual environment for Baran (Python 3.9)..."
python3.9 -m venv venv
source venv/bin/activate
pip install wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install raha -i https://pypi.tuna.tsinghua.edu.cn/simple
deactivate
echo "Baran environment setup complete."

# Set the PYTHONPATH environment variable for Baran
echo 'export PYTHONPATH=/root/AutoMLClustering' >> ~/.bashrc
source ~/.bashrc
echo "Baran's PYTHONPATH has been set to include ${WORK_DIR} in .bashrc."
