#!/bin/bash

# Convert to absolute path before changing directory
test_file=$(realpath "$1")
cd ..

python /home/shin/y3/y3/compilers/project/bxc.py "$test_file"
bxfile=$(basename "$test_file" .bx)
gcc -g -o "$bxfile.exe" "$bxfile.s" bxruntime.c
./"$bxfile.exe"

rm -f "$bxfile.exe" "$bxfile.s"
