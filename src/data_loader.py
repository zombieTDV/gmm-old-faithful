"""
data_loader.py — Manual CSV file parser using Python file I/O.

Loads the Old Faithful dataset from CSV format without any external libraries.
Parses each line manually, handling edge cases (empty lines, invalid values).
"""


def load_csv(filepath):
    """
    Load a CSV file and return data as a list of [float, float] rows.
    
    Parses the CSV manually using open() and string splitting.
    Skips the header row and any rows with invalid/missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        list: List of [eruptions, waiting] pairs as floats.
    """
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Skip header (first line)
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        parts = line.split(',')
        
        # Expect exactly 2 columns: eruptions, waiting
        if len(parts) != 2:
            print(f"  [SKIP] Row {i}: expected 2 columns, got {len(parts)}")
            continue
        
        try:
            eruptions = float(parts[0])
            waiting = float(parts[1])
        except ValueError:
            print(f"  [SKIP] Row {i}: non-numeric value '{parts}'")
            continue
        
        data.append([eruptions, waiting])
    
    print(f"  Loaded {len(data)} valid rows from {filepath}")
    return data
