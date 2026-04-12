#!/bin/bash
# MeshML Python Worker - Automatic Installer
# Usage: curl -sSL https://meshml.io/install.sh | bash

set -e

echo "====================================="
echo "   MeshML Worker Installer v1.0"
echo "====================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    CYGWIN*)    PLATFORM=Windows;;
    MINGW*)     PLATFORM=Windows;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac

echo -e "${GREEN}Detected platform: ${PLATFORM}${NC}"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.10 or higher and try again"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}Found Python ${PYTHON_VERSION}${NC}"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip3 not found, installing...${NC}"
    python3 -m ensurepip --upgrade
fi

# Installation directory
INSTALL_DIR="${HOME}/.meshml"
echo ""
echo "Installation directory: ${INSTALL_DIR}"

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Download worker code (in production, this would pull from GitHub releases)
echo ""
echo "Downloading MeshML worker..."
# git clone --depth 1 https://github.com/navnithb2001/MeshML.git
# For now, assume we're in the repo
if [ ! -d "workers/python-worker" ]; then
    echo -e "${RED}Error: Worker code not found${NC}"
    echo "This script should be run from the MeshML repository"
    exit 1
fi

WORKER_DIR="${INSTALL_DIR}/worker"
mkdir -p "${WORKER_DIR}"

# Copy worker files
echo "Setting up worker..."
cp -r workers/python-worker/* "${WORKER_DIR}/"
cd "${WORKER_DIR}"

# Install dependencies
echo ""
echo "Installing dependencies..."
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install --no-dev
else
    echo "Using pip..."
    pip3 install --user -r requirements.txt
fi

# Interactive configuration
echo ""
echo "====================================="
echo "   Worker Configuration"
echo "====================================="
echo ""

# Get user input
read -p "Enter your email or username: " USER_EMAIL
read -p "Enter worker name (or press Enter for auto-generated): " WORKER_NAME
read -p "Enter Parameter Server URL [http://localhost:50051]: " PARAM_SERVER_URL
PARAM_SERVER_URL=${PARAM_SERVER_URL:-http://localhost:50051}

# Generate worker ID if not provided
if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="worker-$(hostname)-$(date +%s)"
fi

# Detect GPU
HAS_GPU="false"
if command -v nvidia-smi &> /dev/null; then
    HAS_GPU="true"
    echo -e "${GREEN}NVIDIA GPU detected${NC}"
elif [ "$PLATFORM" = "Mac" ]; then
    # Check for Apple Silicon
    if [ "$(uname -m)" = "arm64" ]; then
        HAS_GPU="mps"
        echo -e "${GREEN}Apple Silicon detected (MPS support)${NC}"
    fi
fi

# Create configuration file
echo ""
echo "Creating configuration..."
cat > "${WORKER_DIR}/config.yaml" <<EOF
# MeshML Worker Configuration
# Auto-generated on $(date)

worker:
  worker_id: "${WORKER_NAME}"
  user_email: "${USER_EMAIL}"
  device: "cuda"  # Will auto-detect at runtime
  max_retries: 3
  retry_delay: 5

parameter_server:
  grpc_url: "${PARAM_SERVER_URL}"

storage:
  base_dir: "${HOME}/.meshml/data"
  models_dir: "${HOME}/.meshml/models"
  data_dir: "${HOME}/.meshml/datasets"
  checkpoints_dir: "${HOME}/.meshml/checkpoints"

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10
  use_mixed_precision: ${HAS_GPU}
EOF

echo -e "${GREEN}Configuration saved to ${WORKER_DIR}/config.yaml${NC}"

# Create launcher script
echo ""
echo "Creating launcher..."
cat > "${INSTALL_DIR}/meshml-worker" <<'EOF'
#!/bin/bash
# MeshML Worker Launcher
WORKER_DIR="${HOME}/.meshml/worker"
cd "${WORKER_DIR}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "mesh.venv" ]; then
    source mesh.venv/bin/activate
fi

# Run worker
python3 -m meshml_worker.cli train --config config.yaml "$@"
EOF

chmod +x "${INSTALL_DIR}/meshml-worker"

# Add to PATH
echo ""
echo "Adding to PATH..."
SHELL_RC=""
if [ -f "${HOME}/.bashrc" ]; then
    SHELL_RC="${HOME}/.bashrc"
elif [ -f "${HOME}/.zshrc" ]; then
    SHELL_RC="${HOME}/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "meshml-worker" "$SHELL_RC"; then
        echo "" >> "$SHELL_RC"
        echo "# MeshML Worker" >> "$SHELL_RC"
        echo "export PATH=\"\$PATH:${INSTALL_DIR}\"" >> "$SHELL_RC"
        echo -e "${GREEN}Added to ${SHELL_RC}${NC}"
    fi
fi

# Create systemd service (Linux only)
if [ "$PLATFORM" = "Linux" ] && command -v systemctl &> /dev/null; then
    echo ""
    read -p "Install as system service (auto-start on boot)? [y/N]: " INSTALL_SERVICE
    if [[ $INSTALL_SERVICE =~ ^[Yy]$ ]]; then
        sudo tee /etc/systemd/system/meshml-worker.service > /dev/null <<EOF
[Unit]
Description=MeshML Distributed Training Worker
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${WORKER_DIR}
ExecStart=${INSTALL_DIR}/meshml-worker
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable meshml-worker
        echo -e "${GREEN}Service installed. Start with: sudo systemctl start meshml-worker${NC}"
    fi
fi

# Create LaunchAgent (macOS only)
if [ "$PLATFORM" = "Mac" ]; then
    echo ""
    read -p "Install as LaunchAgent (auto-start on login)? [y/N]: " INSTALL_AGENT
    if [[ $INSTALL_AGENT =~ ^[Yy]$ ]]; then
        PLIST="${HOME}/Library/LaunchAgents/com.meshml.worker.plist"
        cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.meshml.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/meshml-worker</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${HOME}/.meshml/worker.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/.meshml/worker.error.log</string>
</dict>
</plist>
EOF
        launchctl load "$PLIST"
        echo -e "${GREEN}LaunchAgent installed. Worker will start on login.${NC}"
    fi
fi

# Final instructions
echo ""
echo "====================================="
echo "   Installation Complete! 🎉"
echo "====================================="
echo ""
echo "Worker installed to: ${INSTALL_DIR}"
echo "Configuration file: ${WORKER_DIR}/config.yaml"
echo ""

# Optional: Interactive registration
echo -e "${YELLOW}Would you like to register your worker and join a group now? [Y/n]${NC}"
read -r register_choice
register_choice=${register_choice:-Y}

if [[ "$register_choice" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting interactive registration..."
    cd "${WORKER_DIR}"
    python3 -c "
import sys
sys.path.insert(0, '.')
from meshml_worker.config import load_config
from meshml_worker.registration import interactive_registration

config = load_config('config.yaml')
interactive_registration(config)
    " || echo -e "${YELLOW}Registration can be completed later${NC}"
    echo ""
fi

echo ""
echo -e "${GREEN}To start the worker:${NC}"
echo "  ${INSTALL_DIR}/meshml-worker"
echo ""
echo "Or reload your shell and run:"
echo "  meshml-worker"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
if [[ ! "$register_choice" =~ ^[Yy]$ ]]; then
    echo "1. Register worker: meshml-worker register"
    echo "2. Join a training group"
else
    echo "1. Start the worker: meshml-worker"
    echo "2. Monitor training progress"
fi
echo "3. Configure Parameter Server URL if needed (config.yaml)"
echo ""
echo "For help: meshml-worker --help"
echo "Documentation: https://docs.meshml.io"
echo ""
