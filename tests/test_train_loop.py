import subprocess
import os
import sys

def test_train_script_runs():
    # Only test if we can import and run the argparse logic without crashing on syntax errors.
    # We can't fully run it if data is missing, but we can verify imports work.
    
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(proj_root, 'src', 'train.py')
    
    # Run with --help to verify imports and argparse
    result = subprocess.run([sys.executable, script_path, '--help'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'usage: train.py' in result.stdout
