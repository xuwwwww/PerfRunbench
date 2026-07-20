# Optional PyTorch Synthetic Training Example

This example is optional and is not used by the default test suite. It imports
PyTorch only inside the example workload and exits with a clear skip message
when `torch` is unavailable.

It uses synthetic data only and does not download datasets.
