#!/bin/bash
# Setup script for Semantic Proximity Analyzer deployment

set -e

echo "🚀 Semantic Proximity Analyzer - GitHub Setup"
echo "=============================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "streamlit_app_production.py" ]; then
    echo "❌ streamlit_app_production.py not found. Run this script from the project root."
    exit 1
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git config user.name "Semantic Analyzer" || true
    git config user.email "semantic@analyzer.local" || true
else
    echo "✅ Git repository already initialized"
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Semantic Proximity Analyzer with Streamlit Cloud deployment" || echo "⚠️ Nothing new to commit"

# Get repo status
echo ""
echo "✅ Repository ready!"
echo ""
echo "📋 Next Steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "2. Name it: semantic-proximity-analyzer"
echo "3. Copy the commands below:"
echo ""
echo "   git remote add origin https://github.com/YOUR-USERNAME/semantic-proximity-analyzer.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Go to https://share.streamlit.io and deploy!"
echo ""
