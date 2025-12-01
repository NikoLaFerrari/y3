#!/bin/bash

# Clean up previous builds and stale PLY tables
rm -f parsetab.py lextab.py
rm -f examples/*.exe examples/*.s examples/*.tac.json

# Loop through all .bx files
for src in examples/*.bx; do
    base="${src%.*}"
    echo "==================================================="
    echo "Testing: $src"

    # 1. Compile (Generate .s)
    python3 bxc.py "$src"
    if [ $? -ne 0 ]; then
        echo "❌ Compilation failed."
        continue
    fi

    # Verify the .s file exists
    if [ ! -f "$base.s" ]; then
        echo "❌ Error: $base.s was not created!"
        continue
    fi

    # 2. Assemble and Link
    gcc -g -o "$base.exe" "$base.s" bx_runtime.c
    if [ $? -ne 0 ]; then
        echo "❌ Linking failed."
        continue
    fi

    # 3. Run
    echo "Output:"
    ./"$base.exe"
    echo "✅ Done."
done
