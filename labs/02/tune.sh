#!/bin/bash

for args in --epsilon={0.1,0.05,0.01,0.005,0.001}\ --epsilon_final=0.001\ --gamma={0.1,0.2,0.4,0.7,0.8,0.9,0.95,0.99}; do
  qsub -o logs-optimistic/mc-"$args" -N mc ../.venv/bin/python monte_carlo.py $args
done
