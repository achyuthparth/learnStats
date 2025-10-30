"""
Quick Start Script for Wooldridge Econometrics Practice

This script helps you get started with the project setup.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("WOOLDRIDGE ECONOMETRICS - SETUP SCRIPT")
    print("=" * 80)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Install requirements
    print("\n" + "=" * 80)
    print("STEP 1: Installing Required Packages")
    print("=" * 80)
    response = input("\nWould you like to install required packages? (y/n): ")
    
    if response.lower() == 'y':
        try:
            print("\nInstalling packages from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("✗ Error installing packages. Please install manually using:")
            print("  pip install -r requirements.txt")
    
    # GPU Setup
    print("\n" + "=" * 80)
    print("STEP 2: GPU Setup (Optional)")
    print("=" * 80)
    print("\nDo you have a CUDA-capable GPU and want to use it?")
    print("Note: This requires CUDA toolkit and GPU-specific packages.")
    gpu_response = input("Enable GPU support? (y/n): ")
    
    if gpu_response.lower() == 'y':
        print("\nTo enable GPU support:")
        print("1. Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("2. Uncomment GPU packages in requirements.txt")
        print("3. Install GPU packages:")
        print("   pip install cupy-cuda11x  # Replace 11x with your CUDA version")
        print("   pip install cudf-cu11     # RAPIDS cuDF")
        print("4. Set use_gpu = True in config.py")
    else:
        print("✓ Continuing with CPU-only setup (recommended for most Wooldridge exercises)")
    
    # Directory check
    print("\n" + "=" * 80)
    print("STEP 3: Directory Structure")
    print("=" * 80)
    
    directories = {
        'data': Path('data'),
        'notebooks': Path('notebooks'),
        'utils': Path('utils'),
        'outputs': Path('outputs')
    }
    
    print("\nChecking directories:")
    for name, path in directories.items():
        if path.exists():
            print(f"  ✓ {name}/ exists")
        else:
            print(f"  ✗ {name}/ missing")
    
    # Data download instructions
    print("\n" + "=" * 80)
    print("STEP 4: Downloading Datasets")
    print("=" * 80)
    print("\nDatasets are available at:")
    print("https://faculty.utrgv.edu/diego.escobari/teaching/Datasets.html")
    print("\nFor each dataset you need:")
    print("  1. Download the .xls file (contains the data)")
    print("  2. Download the .txt file (contains variable descriptions)")
    print("  3. Save both files to the data/ directory")
    print("\nExample: For Chapter 2, you might need 'wage1.xls' and 'wage1.txt'")
    
    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Download datasets you need from the URL above")
    print("2. Copy notebooks/chapter_template.ipynb to start a new chapter")
    print("3. Open the notebook in Jupyter and begin exercises!")
    print("\nExample commands:")
    print("  # Start Jupyter")
    print("  jupyter notebook")
    print("\n  # Or use VS Code's Jupyter extension (recommended)")
    
    # Test import
    print("\n" + "=" * 80)
    print("STEP 5: Testing Setup")
    print("=" * 80)
    test_response = input("\nWould you like to test the utils module? (y/n): ")
    
    if test_response.lower() == 'y':
        try:
            print("\nImporting econometrics utilities...")
            from utils import (
                load_wooldridge_data,
                run_ols,
                check_gpu_available
            )
            print("✓ Utils module imported successfully!")
            print(f"✓ GPU available: {check_gpu_available()}")
        except ImportError as e:
            print(f"✗ Error importing utils: {e}")
            print("  Make sure you're running this from the Practice directory")
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print("\nYou're ready to start working on Wooldridge exercises!")
    print("Good luck with your econometrics studies! 📊📈")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
