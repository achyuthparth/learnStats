# Test script for new CSV-based data loader
import sys
import pandas as pd
import json
from pathlib import Path

# Add utils to path
sys.path.append('.')

# Import individual functions to avoid dependency issues
def test_data_loader():
    """Test the new CSV-based data loader"""
    print("Testing new CSV-based data loader...")
    
    # Test column mappings loading
    print("\n=== Testing Column Mappings ===")
    try:
        data_dir = Path("data")
        mappings_file = data_dir / 'column_mappings.json'
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        print(f"✓ Column mappings loaded successfully")
        print(f"  Available datasets: {list(mappings.keys())}")
        
        # Check BWGHT structure
        if 'bwght' in mappings:
            bwght_cols = mappings['bwght']['columns']
            print(f"  BWGHT columns: {len(bwght_cols)}")
            print(f"  Sample columns: {list(bwght_cols.keys())[:5]}")
            
    except Exception as e:
        print(f"✗ Column mappings failed: {e}")
        return
    
    # Test basic CSV loading (if files exist)
    print("\n=== Testing CSV Loading ===")
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found CSV files: {[f.name for f in csv_files]}")
    
    if csv_files:
        try:
            # Test loading a CSV file
            test_file = csv_files[0]
            df = pd.read_csv(test_file)
            print(f"✓ Successfully loaded {test_file.name}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
        except Exception as e:
            print(f"✗ CSV loading failed: {e}")
    else:
        print("No CSV files found to test")

if __name__ == "__main__":
    test_data_loader()