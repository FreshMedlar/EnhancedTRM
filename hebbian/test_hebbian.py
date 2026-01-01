
import sys
import os
import torch
import warnings

# Suppress wandb and warnings for clean output
os.environ["WANDB_MODE"] = "offline"
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.getcwd())

# Adjust import path for local file execution 
# We need 'hebbian' in path to import train
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Patching imports if necessary
# Since train.py does 'from model import ...', we need to run it from hebbian dir or fix path
# We will run this script from project root, but trick path.
sys.path.append(os.path.join(os.getcwd(), 'hebbian'))

import hebbian.train as train

def run_test():
    print("Testing Hebbian Training Loop...")
    
    # Mock Constants for Speed
    train.STEPS = 5
    train.POPULATION_SIZE = 4
    train.HIDDEN_SIZE = 128 # Must be > 65
    train.BATCH_SIZE = 2
    train.SEQ_LEN = 5
    train.wandb.init = lambda *args, **kwargs: None
    train.wandb.log = lambda *args, **kwargs: None
    train.wandb.finish = lambda *args, **kwargs: None
    
    try:
        train.main()
        print("SUCCESS: Training loop ran for 5 steps without error.")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
