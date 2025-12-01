#!/bin/bash

# 1. Setup directories
mkdir -p tests

# 2. Create the Runtime (Required for linking)
cat <<EOF > bx_runtime.c
#include <stdio.h>
#include <stdint.h>

void _bx_print_int(int64_t x) {
    printf("%ld\n", x);
}

void _bx_print_bool(int64_t b) {
    printf(b == 0 ? "false\n" : "true\n");
}
EOF

# ==============================================================================
# 3. Generate Test Files
# ==============================================================================

# --- CFG Tests ---

cat <<EOF > tests/cfg_diamond.bx
// Diamond shape CFG
def main() {
  var x = 10 : int;
  if (x > 5) {
    x = x + 1;
  } else {
    x = x - 1;
  }
  print(x);
}
EOF

cat <<EOF > tests/cfg_loop_break.bx
// Loop with break (multiple successors/predecessors)
def main() {
  var i = 0 : int;
  var sum = 0 : int;
  while (i < 10) {
    if (i == 5) { break; }
    sum = sum + i;
    i = i + 1;
  }
  print(sum);
}
EOF

cat <<EOF > tests/cfg_unreachable.bx
// Unreachable Code Elimination
def main() {
  var x = 10 : int;
  if (true) {
    print(1);
    ret;
    // Dead code below
    x = 20; 
    print(x);
  }
  print(0);
}
EOF

# --- Optimization Tests ---

cat <<EOF > tests/opt_dse.bx
// Dead Store Elimination
def main() {
  var x = 0 : int;
  x = 100; // Dead
  x = 200; // Dead
  x = 300; // Live
  print(x);
  x = 400; // Dead
}
EOF

cat <<EOF > tests/opt_copy_prop.bx
// Copy Propagation
def main() {
  var a = 42 : int;
  var b = 0 : int;
  var c = 0 : int;
  b = a; 
  c = b;
  print(c); // Should optimize to use 'a' or '42'
}
EOF

cat <<EOF > tests/opt_dse_loop.bx
// DSE safety inside loops
def main() {
  var i = 0 : int;
  var x = 0 : int;
  while (i < 5) {
    x = x + 1; 
    print(x);
    i = i + 1;
  }
}
EOF

# ==============================================================================
# 4. Run Tests
# ==============================================================================

# Function to run a test
run_test() {
    file=$1
    flags=$2
    desc=$3
    
    base="tests/$(basename "$file" .bx)"
    
    echo "----------------------------------------------------------------"
    echo "TESTING: $file ($desc)"
    echo "FLAGS:   $flags"
    
    # 1. Run Compiler
    python3 bxc1.py "$file" $flags > "$base.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Compile Failed!"
        cat "$base.log"
        return
    fi

    # 2. If dumping CFG, show output
    if [[ "$flags" == *"--dump-cfg"* ]]; then
        echo "📋 CFG Dump:"
        python3 bxc1.py "$file" --dump-cfg | head -n 20
        echo "... (truncated)"
    fi

    # 3. Link and Run (Verification)
    if [ -f "$base.s" ]; then
        gcc -g -o "$base.exe" "$base.s" bx_runtime.c
        if [ $? -eq 0 ]; then
            echo "🚀 Execution Output:"
            ./"$base.exe"
        else
            echo "❌ Link Failed"
        fi
    else
        echo "⚠️  No assembly generated (analysis only mode)"
    fi
}

echo "================== STARTING TEST SUITE =================="

# Run CFG Tests
for f in tests/cfg_*.bx; do
    run_test "$f" "--dump-cfg" "CFG Construction"
done

# Run Optimization Tests
for f in tests/opt_*.bx; do
    # We keep TAC to manually inspect optimization if needed
    run_test "$f" "--keep-tac" "Dataflow Optimization"
done

echo "================== DONE =================="
