#!/bin/bash
    mkdir -p tests_generated/output
    
    for file in tests/examples/*.bx; do #tests_generated/*.bx; do
        echo "Testing $file..."
        base_name=$(basename "$file" .bx)
        
        # 1. Compile with your python compiler
        python3 bxc_new.py "$file"
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Compilation error"
            continue
        fi
        
        # 2. Link with GCC (assuming bxruntime.c is present)
        gcc -no-pie -o "tests/examples/output/$base_name.exe" "tests/examples/$base_name.s" bxruntime.c
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Linking error"
            continue
        fi
        
        # 3. Run the executable
        ./tests/examples/output/$base_name.exe
        if [ $? -ne 0 ]; then
             echo "  [FAIL] Runtime error"
        else
             echo "  [PASS]"
        fi
        echo "--------------------------------"
    done
