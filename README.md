
This repository contains the computational materials associated with the study:

**Integrated production and transportation scheduling in hybrid manufacturing–remanufacturing systems with component reprocessing**

The repository provides:
- the 3,576 scheduling instances used in the computational study;
- the Python implementation of the analytical lower bound; and
- the Python implementations of the adaptive large neighbourhood search (ALNS) and multi-neighbourhood simulated annealing (MNSA) algorithms.

## Repository structure

```text
Main/
├── Instances/
│   ├── Phase_I_small/      # 4-, 6-, and 8-job instances
│   ├── Phase_II_medium/    # 10- and 15-job instances
│   └── Phase_III_large/    # 20-, 50-, 100-, and 200-job instances
├── Lower_bounds.py
└── Metaheuristics.py
