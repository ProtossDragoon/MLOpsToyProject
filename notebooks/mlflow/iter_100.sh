#!/bin/bash
for i in {1..100} ; do
    echo "Number $i: $(date +%Y-%m-%d-%H:%M:%S)"
    python3 mlops.py $i
done
