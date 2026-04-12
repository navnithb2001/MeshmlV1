#!/bin/bash
set -e

# ==========================================================
# MeshML Dashboard Firebase Deployment Script 
# ==========================================================

echo "Have you logged into Firebase CLI and run 'firebase init hosting' in the dashboard directory? (y/n)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Please run those setup steps first and re-run this script."
    exit 1
fi

echo "[1/3] Navigating to dashboard directory..."
# This assumes the script is run from the root of the MeshML repository
cd dashboard

echo "[2/3] Building the production assets..."
npm install
npm run build 

echo "[3/3] Deploying to Firebase Hosting..."
if ! command -v firebase &> /dev/null; then
    echo "Firebase CLI could not be found. Please install it globally with: npm install -g firebase-tools"
    exit 1
fi

# Deploy only the static hosting assets (the dist/ folder)
firebase deploy --only hosting

echo "=========================================================="
echo "DASHBOARD DEPLOYMENT COMPLETE!"
echo "=========================================================="
