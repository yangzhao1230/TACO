#!/bin/bash

cells=("sknsh" "k562" "hepg2")

levels=("mbo")

tfbs_lambdas=(0.0 0.01 0.1)

for cell in "${cells[@]}"; do
    for level in "${levels[@]}"; do
        for tfbs_lambda in "${tfbs_lambdas[@]}"; do
            echo "Executing cell: $cell, level: $level, tfbs_lambda: $tfbs_lambda"
            python aggregate_mbo.py --cell "$cell" --level "$level" --tfbs_lambda "$tfbs_lambda"
        done
    done
done

echo "All combinations executed successfully."
