#!/bin/bash
set -ex

DIR="$(readlink -f $(dirname $0))"
cd "$DIR"

rm -f tmp/*

runner=sge

results_dir=${1:-reinforce}

if [ -d $results_dir ]; then
  echo "Directory $results_dir already exists, remove it first."
  exit 1
fi

~arnold/.venvs/bopt/bin/bopt init \
        --param "batch_size:int:4:128" \
        --param "gamma:logscale_float:0.5:1.0" \
        --param "hidden_layer:int:2:128" \
        --param "learning_rate:logscale_float:1e-6:1e-1" \
        -C $results_dir \
        --qsub=-q --qsub=cpu-troja.q --qsub=-pe --qsub=smp --qsub=16 \
        --qsub=-l --qsub=mem_free=8G,act_mem_free=8G,h_vmem=12G \
        --runner $runner \
        --ard=1 --gamma-prior=1 --gamma-a=1.0 --gamma-b=0.001 \
        $DIR/reinforce.sh

        # --qsub=-l --qsub="mem_free=8G,act_mem_free=8G,h_vmem=12G" \
