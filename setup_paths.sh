#!/bin/bash

# Detect the path to NVFlare binary
NVFLARE_PATH=$(which nvflare)

# Check if NVFlare is installed
if [ -z "$NVFLARE_PATH" ]; then
    echo "Error: nvflare is not installed or not found in the system PATH."
    exit 1
fi

# Automatically detect the path of the MeshDist_nvflare repo's app/code folder
REPO_ROOT=$(pwd)
PYTHONPATH_TO_ADD="$REPO_ROOT/app/code/"

# Detect the right shell configuration file
if [ -f ~/.bashrc ]; then
    SHELL_CONFIG_FILE=~/.bashrc
elif [ -f ~/.bash_profile ]; then
    SHELL_CONFIG_FILE=~/.bash_profile
elif [ -f ~/.zshrc ]; then
    SHELL_CONFIG_FILE=~/.zshrc
else
    echo "Error: Could not find .bashrc, .bash_profile, or .zshrc. Please manually configure your shell environment."
    exit 1
fi

# Add NVFlare PATH to the detected shell configuration file if it's not already there
if grep -q "$NVFLARE_PATH" "$SHELL_CONFIG_FILE"; then
    echo "NVFlare PATH already added to $SHELL_CONFIG_FILE."
else
    echo "Adding NVFlare PATH to $SHELL_CONFIG_FILE..."
    echo "export PATH=$(dirname $NVFLARE_PATH):\$PATH" >> "$SHELL_CONFIG_FILE"
fi

# Add PYTHONPATH to the detected shell configuration file if it's not already there
if grep -q "$PYTHONPATH_TO_ADD" "$SHELL_CONFIG_FILE"; then
    echo "PYTHONPATH already added to $SHELL_CONFIG_FILE."
else
    echo "Adding PYTHONPATH to $SHELL_CONFIG_FILE..."
    echo "export PYTHONPATH=\$PYTHONPATH:$PYTHONPATH_TO_ADD" >> "$SHELL_CONFIG_FILE"
fi

# Apply the changes immediately in the current session
source "$SHELL_CONFIG_FILE"

echo "Environment setup complete. Changes applied to $SHELL_CONFIG_FILE."

