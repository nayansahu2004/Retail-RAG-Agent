#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 EC2 instance (t3.small or larger).
# Usage (on the server):  bash ec2-setup.sh
set -euo pipefail

echo "==> Installing Docker Engine + Compose plugin (official repository)"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

echo "==> Adding 2 GB swap (protects small instances from out-of-memory kills)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==> Creating app directory"
mkdir -p ~/retail-rag-agent
echo "Next steps:"
echo "  1. Create ~/retail-rag-agent/.env with GROQ_API_KEY and GEMINI_API_KEY (see .env.example)"
echo "  2. Log out and back in so the docker group applies"
echo "  3. Push to main: GitHub Actions copies docker-compose.prod.yml and starts the app"