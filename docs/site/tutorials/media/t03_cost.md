<!-- written by docs/site/tutorial_media_render.py from the captured data; do not edit -->
```{table} Size of each eigenproblem and wall time of its dense QZ solve, 7 timings each (median and range), from one run on AMD EPYC 9654 96-Core Processor with 2 threads (NumPy 2.5.3, SciPy 1.17.1). Halving the unknowns made the solve 5.0 times faster here; the n³ operation count bounds the saving at 8.
:name: t03-cost

| problem | unknowns n | QZ time, median (ms) | range (ms) |
|:--|--:|--:|--:|
| full space | 268 | 74 | 70 – 75 |
| onset class | 134 | 15 | 15 – 67 |
| other class | 134 | 15 | 15 – 60 |
```
