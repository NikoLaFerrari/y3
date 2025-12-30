    .data
    .globl glob_a
glob_a:
    .zero 8
    .globl glob_b
glob_b:
    .zero 8
    .globl glob_c
glob_c:
    .zero 8
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_c(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -32(%rbp)
    movq glob_b(%rip), %rcx
    movq %rcx, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rcx
    movq %rcx, glob_a(%rip)
    movq glob_a(%rip), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_a(%rip), %rcx
    movq %rcx, -64(%rbp)
    movq $2, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rcx
    movq %rcx, glob_b(%rip)
    movq glob_b(%rip), %rcx
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_b(%rip), %rcx
    movq %rcx, -96(%rbp)
    movq glob_a(%rip), %rcx
    movq %rcx, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -112(%rbp)
    movq -112(%rbp), %rcx
    movq %rcx, glob_c(%rip)
    movq glob_c(%rip), %rcx
    movq %rcx, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
