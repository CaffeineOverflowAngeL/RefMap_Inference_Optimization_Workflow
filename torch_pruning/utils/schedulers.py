import numpy as np

def alternate_pruning_lr_schedule(init_lr, total_steps):
    """
    Create a learning rate schedule that first uses linear decay and then cosine annealing.
    """
    half_steps = total_steps // 2
    linear_decay_lr = np.linspace(init_lr, init_lr * 0.1, half_steps)
    
    # Cosine annealing schedule
    cosine_lr = init_lr * 0.1 + 0.9 * init_lr * (1 + np.cos(np.pi * np.arange(half_steps) / half_steps)) / 2
    
    return np.concatenate([linear_decay_lr, cosine_lr]).tolist()

def alterante_cosine_lr_schedule(init_lr, min_lr, total_steps):
    """
    Create a cosine learning rate schedule from init_lr to min_lr over total_steps.
    
    Parameters:
    - init_lr: initial learning rate.
    - min_lr: minimum learning rate.
    - total_steps: total number of steps or iterations.
    
    Returns:
    - List containing the learning rate for each step.
    """
    # Compute the difference between initial and minimum learning rates
    lr_diff = init_lr - min_lr
    
    # Compute the cosine annealed learning rates
    lr_schedule = [
        min_lr + 0.5 * lr_diff * (1 + np.cos(np.pi * t / total_steps))
        for t in range(total_steps)
    ]

    # Format each value to have 4 decimal places of precision
    lr_schedule = [round(lr, 4) for lr in lr_schedule]
    
    return lr_schedule