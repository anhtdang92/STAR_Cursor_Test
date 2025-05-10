#!/bin/bash

echo "Starting STAR Diffusion Model Setup..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Make the script executable
chmod +x setup.py

# Run the setup script
python3 setup.py

# If setup was successful, activate the environment and start the server
if [ $? -eq 0 ]; then
    echo
    echo "Setup completed successfully!"
    echo "Starting the backend server..."
    echo
    source venv/bin/activate
    python backend/server.py
else
    echo
    echo "Setup failed. Please check the error messages above."
fi 