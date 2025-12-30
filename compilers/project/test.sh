#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_file> <dir_with_bx_files>"
    exit 1
fi

PYFILE="$1"
BXDIR="$2"

shopt -s nullglob

total=0
success=0
fail=0

LOGFILE="test_log.txt"
echo "BX Compiler Test Log" > "$LOGFILE"
echo "====================" >> "$LOGFILE"
echo >> "$LOGFILE"

echo "Successful Tests:" >> "$LOGFILE"
echo "-----------------" >> "$LOGFILE"

SUCCESS_SECTION_LINE=$(wc -l < "$LOGFILE")

echo >> "$LOGFILE"
echo "Failed Tests:" >> "$LOGFILE"
echo "-------------" >> "$LOGFILE"

FAIL_SECTION_LINE=$(wc -l < "$LOGFILE")

# counters for logging placement
success_list=""
fail_list=""

for bx in "$BXDIR"/*.bx; do
    total=$((total + 1))

    echo "=============================="
    echo "Running: $PYFILE on $bx"
    echo "------------------------------"

    # capture python output + status
    output=$(python "$PYFILE" "$bx" 2>&1)
    status=$?

    base="${bx%.bx}"
    sfile="${base}.s"

    if [ $status -ne 0 ]; then
        echo "❌ Compiler returned non-zero exit code ($status)."
        echo "❌ FAILED: $bx" >> "$LOGFILE"
        echo "Error message:" >> "$LOGFILE"
        echo "$output" >> "$LOGFILE"
        echo >> "$LOGFILE"

        fail=$((fail + 1))
        echo
        continue
    fi

    if [ -f "$sfile" ]; then
        echo "✅ SUCCESS: '$sfile' generated."
        echo "✔ $bx" >> "$LOGFILE"
        success=$((success + 1))
    else
        echo "❌ FAILURE: '$sfile' NOT generated."

        echo "❌ FAILED: $bx" >> "$LOGFILE"
        echo "Reason: .s output file not produced" >> "$LOGFILE"
        echo "Compiler output:" >> "$LOGFILE"
        echo "$output" >> "$LOGFILE"
        echo >> "$LOGFILE"

        fail=$((fail + 1))
    fi

    echo
done

echo "=============================="
echo "          SUMMARY"
echo "=============================="
echo "Total tests:   $total"
echo "Succeeded:     $success"
echo "Failed:        $fail"
echo "=============================="
echo
echo "Log file written to: $LOGFILE"

echo >> "$LOGFILE"
echo "====================" >> "$LOGFILE"
echo "SUMMARY" >> "$LOGFILE"
echo "====================" >> "$LOGFILE"
echo "Total tests:   $total" >> "$LOGFILE"
echo "Succeeded:     $success" >> "$LOGFILE"
echo "Failed:        $fail" >> "$LOGFILE"

