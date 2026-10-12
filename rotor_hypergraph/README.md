# Rotor Fault Time-Series Characterisation

Code for the paper: *Saturated Symbolic Entropy Phase-Space Reconstruction and
NI-Louvain Driven Multi-Resolution Hypergraph Modelling for Rotor Fault
Time-Series Characterisation* .

## Layout

```
paper1_rotor/
├── src/            Python source (experiments, figures, tables)
├── results/        JSON results produced by the experiments
├── figures/        PDF figures
├── tables/         TeX tables
└── manuscript/     LaTeX manuscript
```

## Setup

```bash
pip install numpy scipy matplotlib networkx community-louvain vmdpy scikit-learn
export ROTOR_DATA_ROOT=/path/to/raw/data   # contains 180data_*.mat, 360rotor_*.mat, MFPT/
cd src
python exp1_psr.py        # phase-space reconstruction benchmarks
python exp3_hypergraph.py # hypergraph topology
python exp5_ablation.py   # ablation
python exp7_classification.py
python exp10_downstream.py
python exp13_extreme.py   # extreme noise sweep
python exp14_fixedtau.py  # fixed-tau separability sweep
python make_tables.py
python make_figures.py
```

The MFPT bearing dataset is public; the measured rotor records are
proprietary and are not redistributed here.

## License

MIT.
