for f in examples/*.bx
    set base (basename $f .bx)
    echo "=== $base ==="

    # compile BX -> .s (+ optional tac)
    python bxc1.py $f >/dev/null 2>examples/$base.compile.err
    if test $status -ne 0
        echo "❌ compile failed (see examples/$base.compile.err)"
        echo
        continue
    end

    # assemble/link
    gcc -no-pie -g -o examples/$base.exe examples/$base.s bx_runtime.c \
        >/dev/null 2>examples/$base.link.err
    if test $status -ne 0
        echo "❌ link failed (see examples/$base.link.err)"
        echo
        continue
    end

    # run
    ./examples/$base.exe > examples/$base.out 2>examples/$base.run.err
    if test $status -ne 0
        echo "❌ runtime failed (see examples/$base.run.err)"
        echo
        continue
    end

    echo "Output:"
    cat examples/$base.out
    echo
end

