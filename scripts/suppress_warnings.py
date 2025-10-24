#!/usr/bin/env python3
"""
Suppress common deprecation warnings during training
Add this to the top of train scripts to clean up output
"""

import warnings

# Suppress torch_dtype deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch_dtype.*is deprecated.*Use.*dtype.*instead.*",
    category=UserWarning,
)

# Suppress other common warnings
warnings.filterwarnings(
    "ignore",
    message=".*PYTORCH_CUDA_ALLOC_CONF.*deprecated.*",
    category=UserWarning,
)

print("Warnings suppressed")
