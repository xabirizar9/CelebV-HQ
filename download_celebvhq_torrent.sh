#!/bin/bash

# Script to download CelebV-HQ dataset using transmission-cli
# Make sure to run: chmod +x download_celebvhq_torrent.sh before running

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define variables
MAGNET_LINK="magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&dn=CelebV-HQ&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=https%3A%2F%2Fipv6.academictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80%2Fannounce"
DOWNLOAD_DIR="/mnt/disks/CelebV-HQ"

# Check if download directory exists, create if not
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo -e "${YELLOW}Creating download directory at $DOWNLOAD_DIR${NC}"
    mkdir -p "$DOWNLOAD_DIR"
fi

# Function to check if a command is installed
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed.${NC}"
        echo -e "${YELLOW}Please install $1 using your package manager:${NC}"
        echo -e "${YELLOW}For Ubuntu/Debian: sudo apt-get install $2${NC}"
        echo -e "${YELLOW}For Fedora: sudo dnf install $2${NC}"
        echo -e "${YELLOW}For Arch: sudo pacman -S $2${NC}"
        return 1
    fi
    return 0
}

# Try different torrent clients
download_with_transmission() {
    echo -e "${GREEN}Starting download with transmission-cli...${NC}"
    transmission-cli "$MAGNET_LINK" -w "$DOWNLOAD_DIR"
}

download_with_aria2() {
    echo -e "${GREEN}Starting download with aria2c...${NC}"
    aria2c --dir="$DOWNLOAD_DIR" --seed-time=0 "$MAGNET_LINK"
}

download_with_qbittorrent() {
    echo -e "${GREEN}Starting download with qBittorrent...${NC}"
    qbittorrent-nox --save-path="$DOWNLOAD_DIR" "$MAGNET_LINK"
}

# Main function
main() {
    echo -e "${GREEN}=== CelebV-HQ Dataset Downloader ===${NC}"
    echo -e "${YELLOW}This script will download the CelebV-HQ dataset (approx. 242GB) to $DOWNLOAD_DIR${NC}"
    echo -e "${YELLOW}The download may take a long time depending on your internet connection.${NC}"
    echo ""
    
    # Check for transmission-cli
    if check_command "transmission-cli" "transmission-cli"; then
        download_with_transmission
        return 0
    fi
    
    # If transmission is not available, try aria2c
    if check_command "aria2c" "aria2"; then
        download_with_aria2
        return 0
    fi
    
    # If aria2c is not available, try qbittorrent
    if check_command "qbittorrent-nox" "qbittorrent-nox"; then
        download_with_qbittorrent
        return 0
    fi
    
    # If we reached here, no torrent client is available
    echo -e "${RED}Error: No supported torrent client found.${NC}"
    echo -e "${YELLOW}Please install one of the following:${NC}"
    echo -e "${YELLOW}- transmission-cli${NC}"
    echo -e "${YELLOW}- aria2${NC}"
    echo -e "${YELLOW}- qbittorrent-nox${NC}"
    return 1
}

# Run the main function
main

echo -e "${GREEN}Script completed.${NC}"
echo -e "${YELLOW}If the download was started successfully, it will continue in the background.${NC}"
echo -e "${YELLOW}The dataset will be downloaded to $DOWNLOAD_DIR${NC}" 