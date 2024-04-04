"""
    Function to be mapped to tokenised text to create labels data set.
    Applied to each element then return some ordering of the new updated tensor
"""

def standard_autoregressive_label(input_tensor):
    return input_tensor[:-1], input_tensor[1:]