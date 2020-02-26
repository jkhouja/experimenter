#!/bin/bash

PYTHON_FILE=$1
DIRECTORY=$2

for i in $(ls  $DIRECTORY); do
    echo "Running file: $DIRECTORY$i with python file $PYTHON_FILE"
    python $PYTHON_FILE --config_file $DIRECTORY$i
    echo "====================================================================================="
done

