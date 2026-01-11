import numpy as np

def calculate_shannon_entropy(grid):
    """
    Measures the 'Informational Diversity' of the PCC system.
    H = 0: Monoculture (System Death/Extinction)
    H = 1.58 (log2(3)): Perfect Balance
    """
    # Get counts of P, Co, and Ch
    _, counts = np.unique(grid, return_counts=True)
    
    # Calculate probabilities
    probs = counts / counts.sum()
    
    # Shannon Entropy formula
    entropy = -np.sum(probs * np.log2(probs + 1e-9)) # 1e-9 prevents log(0)
    return entropy

def calculate_volatility(history):
    """
    Measures how fast the 'Power Balance' is shifting.
    High volatility = High Chaos dominance.
    Low volatility = High Control dominance.
    """
    if len(history) < 2:
        return 0
    return np.std(np.diff(history, axis=0))
