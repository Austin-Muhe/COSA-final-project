| Method | Test MSE | Test MAE | MSE improve vs Base | MAE improve vs Base | Notes |
|---|---:|---:|---:|---:|---|
| Base frozen backbone | 0.195218 | 0.234493 | 0.00% | 0.00% | No adapter |
| DynamicRegimeAdapter untrained | 0.195219 | 0.234494 | -0.00% | -0.00% | No test-time backprop |
| DynamicRegimeAdapter meta-trained | 0.191855 | 0.243863 | 1.72% | -4.00% | No test-time backprop; validation meta-trained |
| COSA original | 0.191272 | 0.237397 | 2.02% | -1.24% | Test-time backprop, 492 adaptation steps |
| COSA+ | 0.191641 | 0.237735 | 1.83% | -1.38% | Test-time backprop, 492 adaptation steps |
