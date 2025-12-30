#!/bin/bash

# Regression test script - verifies that all files in regression/ fail to compile
# Each test should exit with a non-zero exit code
cd ..
REGRESSION_DIR="tests/regression"
COMPILER="python3 bxc_new.py"
PASSED=0
FAILED=0
TOTAL=0

echo "Running regression tests..."
echo "=============================="
echo ""

# Find all .bx files in regression directory (not in subdirectories like lab2/)
for test_file in "$REGRESSION_DIR"/*.bx; do
    if [ -f "$test_file" ]; then
        TOTAL=$((TOTAL + 1))
        filename=$(basename "$test_file")
        
        # Try to compile the file, redirect output to /dev/null
        $COMPILER "$test_file" > /dev/null 2>&1
        exit_code=$?
        
        bxfile=$(basename "$test_file" .bx)
        rm -f "$bxfile.s" "$bxfile.exe"
        
        if [ $exit_code -ne 0 ]; then
            echo "✓ PASS: $filename (correctly failed with exit code $exit_code)"
            PASSED=$((PASSED + 1))
        else
            echo "✗ FAIL: $filename (should have failed but succeeded!)"
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo ""
echo "=============================="
echo "Results: $PASSED/$TOTAL passed"

if [ $FAILED -eq 0 ]; then
    echo "All regression tests passed! ✓"
    exit 0
else
    echo "$FAILED tests failed! ✗"
    exit 1
fi
