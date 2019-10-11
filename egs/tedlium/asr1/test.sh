#!/bin/bash
echo "Testing." > test.out
echo "PATH:" >> test.out
echo $PATH >> test.out
echo "python3:" >> test.out
echo $(which python3) >> test.out
