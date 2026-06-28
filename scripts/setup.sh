#!/bin/bash
# SeismicLab — Quick Setup
set -e

echo "=== SeismicLab Setup ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install Python 3.10+"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Download dataset from HuggingFace
echo "Downloading dataset from HuggingFace..."
python3 scripts/download_data.py

# Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template — edit it to add your API keys"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the server:"
echo "  python3 server.py"
echo ""
echo "Then open http://localhost:8000 in your browser"
echo ""
echo "Optional: add API keys to .env for full data source coverage"
echo "  NASA_API_KEY  — solar/CME data (free at api.nasa.gov)"
echo "  FIRMS_API_KEY — thermal anomalies (free at firms.modaps.eosdis.nasa.gov)"
