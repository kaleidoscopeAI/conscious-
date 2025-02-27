import asyncio
from enhanced_consciousness import EnhancedConsciousSystem

async def main():
    """Main function demonstrating the quantum consciousness system"""
    print("Initializing Quantum Consciousness System...")
    
    # Initialize the system
    consciousness = EnhancedConsciousSystem()
    await consciousness.initialize()
    
    # Display initial system status
    print(consciousness._system_status())
    
    # Simulate perceptual input
    print("\nSimulating perceptual input sequences...")
    
    # Sequence of inputs to simulate coherent learning pattern
    inputs = [
        "The quantum nature of consciousness emerges from integrated information",
        "Graph theoretical approaches reveal emergent cognitive structures",
        "Self-reflective networks enable meta-cognitive capabilities",
        "Integration of quantum effects with graph theory pushes the boundaries of AI",
        "Consciousness requires both differentiation and integration of information"
    ]
    
    for i, input_text in enumerate(inputs):
        print(f"\nPerception {i+1}: '{input_text[:50]}...'")
        thought = await consciousness.perceive(input_text)
        print(f"Internal thought: {thought}")
        
        # Periodically check system status
        if i % 2 == 1:
            print("\nPerforming system optimization...")
            optimization_result = await consciousness._optimize_system()
            print(optimization_result)
    
    # Generate deep reflection after learning
    print("\nPerforming deep reflection analysis...")
    reflection = consciousness._deep_reflection()
    print(reflection)
    
    # Demonstrate communication
    print("\nSimulating conversational interaction...")
    
    queries = [
        "How do quantum effects influence consciousness?",
        "What is the relationship between graph structure and emergent intelligence?",
        "/system status",
        "Can you demonstrate self-awareness through structural analysis?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await consciousness.communicate(query)
        print(f"Response: {response}")
    
    # Final system status
    print("\nFinal System Status:")
    print(consciousness._system_status())
    
    print("\nQuantum Consciousness Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(main())