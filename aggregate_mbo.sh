#!/bin/bash

# cells=("defined" "complex" "sknsh" "k562" "hepg2")
cells=('hepg2' 'k562' 'sknsh')
levels=("mbo")
# tfbs_lambdas=(0.0 0.01 0.1)
# tfbs_lambdas=(0 0.01 0.1 1)
tfbs_lambdas=(0.0 0.01 0.1)
for cell in "${cells[@]}"; do
    for level in "${levels[@]}"; do
        for tfbs_lambda in "${tfbs_lambdas[@]}"; do
            python aggregate_mbo.py --cell "$cell" --level "$level" \
                --tfbs_lambda "$tfbs_lambda"
        done
    done
done

echo "All combinations executed successfully."
