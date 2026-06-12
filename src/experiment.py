import time
from skopt import Optimizer
import numpy as np
from madsci.client import NodeClient

# Configuration parameters
PRUSA_URL = "http://127.0.0.1:2000"
TARGET_HEIGHT = 10.0
BOUNDS = [(10.0, 100.0)]
TOTAL_ITERATIONS = 10

def run_campaign():
    print("Initializing autonomous discovery campaign...")
    
    # Initialize MADSci node client and Gaussian Process optimizer
    prusa_node = NodeClient(PRUSA_URL)
    
    opt = Optimizer(
        dimensions=BOUNDS,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=3,
        random_state=237
    )

    # Main execution loop
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print("\n=========================================")
        print(f"--- Iteration {iteration} / {TOTAL_ITERATIONS} ---")
        
        # Request next dimension from optimizer
        suggested_x = opt.ask()
        ridge_length = float(suggested_x[0])
        
        # Send print command to Prusa node
        print(f"Commanding Prusa node with length: {ridge_length:.2f}mm...")
        prusa_node.wait_for_state("IDLE", timeout=60) 
        
        print_response = prusa_node.action("slice_and_print", {"length": ridge_length})
        print(f"Print status: {print_response.get('status')}")
        
        # Pause for physical hardware transfer
        input("\nACTION REQUIRED: Transfer part to rig, trigger Opentrons OT-2, and press Enter to measure...")
        
        # Send measurement command to Camera node
        print("Commanding Camera node to measure fluid...")
        prusa_node.wait_for_state("IDLE", timeout=60)
        measure_response = prusa_node.action("measure_fluid", {"target_bucket": 3})
        
        # Parse measurement and calculate error relative to target height
        measured_height = measure_response["result"]["height_mm"]
        error_y = abs(TARGET_HEIGHT - measured_height)
        
        # Update surrogate model with physical result
        opt.tell(suggested_x, error_y)
        print(f"Result: {measured_height}mm. Error: {error_y:.2f}mm. Model updated.")

    # Post-campaign analysis
    print("\n=== EXPERIMENT COMPLETE ===")
    best_index = np.argmin(opt.yi)
    print(f"Optimal ridge length: {opt.Xi[best_index][0]:.2f} mm")
    print(f"Minimum error achieved: {opt.yi[best_index]:.2f} mm")

if __name__ == "__main__":
    run_campaign()