#!/usr/bin/env python3
# Quick test script for the Quantum Conscious Cube System

import time
import matplotlib.pyplot as plt
from conscious_cube import ConsciousCube

def main():
    print("Initializing Quantum Conscious Cube...")
    cube = ConsciousCube(dimension=3, resolution=20)  # Smaller resolution for quick test
    
    print("Creating test nodes...")
    nodes = cube.create_nodes(10)  # Create 10 test nodes
    
    print("Creating test connections...")
    # Connect some nodes
    for i in range(len(nodes)-1):
        cube.connect_nodes(nodes[i].id, nodes[i+1].id)
    
    print("Processing for 3 seconds...")
    time.sleep(3)  # Let the system process
    
    # Get system status
    status = cube.get_status()
    print("\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Visualize
    print("\nGenerating visualization...")
    ax = cube.visualize_cube()
    plt.savefig("test_visualization.png")
    print(f"Visualization saved to test_visualization.png")
    
    # Clean up
    cube.stop()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
