---
inclusion: always
---

# Code Style Guidelines for This Project

## Core Principles

1. **Write humanized, readable code** - not enterprise-level over-engineered solutions
2. **Keep it simple** - this is a student project, not production software
3. **Prioritize clarity over cleverness**

## Specific Rules

### Code Length
- Functions should be short and focused (10-30 lines max)
- Scripts should be readable in one sitting (50-150 lines ideal)
- If a script exceeds 200 lines, it's probably over-engineered

### Print Statements
- **NEVER use emojis** in print statements (❌ ✓ ⚠️ 🚀 etc.)
- Use simple text: "Done" not "✓ Done!"
- Use plain markers: "[OK]" or "PASS" instead of checkmarks
- Example: `print("Cleaning complete")` not `print("✓ Cleaning complete!")`

### Function Structure
- Avoid excessive helper functions for simple tasks
- Don't create functions that are only called once
- Inline simple operations instead of abstracting them

### Documentation
- Brief comments where needed, not verbose docstrings
- No excessive header blocks with author, date, description
- A simple `# Does X` comment is enough

### Error Handling
- Don't add try-except blocks unless actually needed
- Let errors fail naturally during development
- Simple `if file exists` checks are fine

### Reporting/Logging
- No elaborate progress bars or formatted tables
- Simple print statements with key information
- No ASCII art, no banner lines with "="*60
- Example: Use `print(f"Processed {len(data)} rows")` not elaborate reports

### Data Processing
- Use pandas/numpy directly without wrapper functions
- Don't create custom classes for simple data operations
- Chain operations when it makes sense

### General Style
- Write code like a smart student would, not like a senior engineer
- Assume the reader understands Python basics
- Don't add "future-proofing" or "scalability" features
- No type hints unless specifically requested
- No elaborate config systems - hardcode reasonable defaults

## Good Example
```python
import pandas as pd

# Load and clean data
df = pd.read_csv('data.csv')
df = df.drop_duplicates()
df = df.dropna()

print(f"Cleaned data: {len(df)} rows")
df.to_csv('cleaned.csv', index=False)
```

## Bad Example (Over-engineered)
```python
"""
Data Cleaning Module
Author: Student
Date: 2024
Description: Comprehensive data cleaning pipeline with validation
"""

import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        raise

# ... 200 more lines of over-engineered code
```

## Summary
Keep it simple, readable, and human. No emojis, no over-engineering, no unnecessary abstraction.
