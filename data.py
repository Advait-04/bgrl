import numpy as np

def get_dataset(size):
    common_length = size

    # Dopamine values
    dvalues_range1 = np.arange(0, 40)
    dvalues_range2 = np.arange(41, 60)
    dvalues_range3 = np.arange(60, 196)
    dopamine_values = np.concatenate((dvalues_range1, dvalues_range2, dvalues_range3))
    dopamine_values = np.tile(dopamine_values, common_length)[:common_length]

    # Acetylcholine values
    acetyl_values = np.zeros(common_length)  # Initialize array to hold acetylcholine values
    for i, dvalue in enumerate(dopamine_values):
        if dvalue >= 0 and dvalue < 40:
            acetyl_values[i] = np.random.randint(3, 10)  # Generate random values between 3 and 6
        elif dvalue >= 40 and dvalue <= 60:
            acetyl_values[i] = np.random.randint(1, 3)  # Generate random values between 1 and 4
        elif dvalue >= 60 and dvalue <= 196:
            acetyl_values[i] = np.random.randint(0, 1)  # Generate random values between 1 and 4

    return [
        dopamine_values, acetyl_values
    ]
