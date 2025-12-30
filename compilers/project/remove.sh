#!/bin/bash

for f in tests/examples/*.bx; do
  # skip if no matching files
  [ -f "$f" ] || continue

  # remove the extern declaration line (handles both `n : int` and `n: int`)
  sed -i '/^extern def print(n *: *int);$/d' "$f"
done

