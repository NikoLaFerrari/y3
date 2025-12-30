#!/bin/bash

# Runs bxcompiler on all .bx files in ../starter/examples and verifies 
# the output matches the expected output file
EXAMPLES_DIR="tests/examples"
COMPILER="python3 bxc_new.py"
PASSED=0
FAILED=0
TOTAL=0

echo "Running example tests..."
echo "=============================="
echo ""

# Use process substitution to avoid subshell issue with counters
while IFS= read -r file; do
    TOTAL=$((TOTAL + 1))
    filename=$(basename "$file")
    expected_output="$EXAMPLES_DIR/$(basename "$file" .bx)_output.txt"
    
    # Check if expected output file exists
    if [ ! -f "$expected_output" ]; then
        echo "⚠ SKIP: $filename (no expected output file)"
        continue
    fi
    
    # Compile the file
    $COMPILER "$file" > /dev/null 2>&1
    # $COMPILER "$file" 2>/dev/null
    compile_exit=$?
    
    bxfile=$(basename "$file" .bx)
    
    if [ $compile_exit -ne 0 ]; then
        echo "✗ FAIL: $filename (compilation failed with exit code $compile_exit)"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Assemble and link
    gcc -g -o "$bxfile.exe" "$bxfile.s" bxruntime.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "✗ FAIL: $filename (assembly/linking failed)"
        FAILED=$((FAILED + 1))
        rm -f "$bxfile.s"
        continue
    fi
    
    # Run and compare output
    if diff -q <(./"$bxfile.exe") "$expected_output" > /dev/null 2>&1; then
        echo "✓ PASS: $filename"
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAIL: $filename (output mismatch)"
        echo "  Expected:"
        head -5 "$expected_output" | sed 's/^/    /'
        echo "  Got:"
        ./"$bxfile.exe" | head -5 | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
    
    # Cleanup
    rm -f "$bxfile.exe" "$bxfile.s"
done < <(find "$EXAMPLES_DIR" -name "*.bx" | sort)

echo ""
echo "=============================="
echo "Results: $PASSED passed, $FAILED failed (total: $TOTAL)"

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "Some tests failed! ✗"
    exit 1
fi
