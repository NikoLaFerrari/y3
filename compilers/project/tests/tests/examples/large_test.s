    .data
    .globl glob_global_a
glob_global_a:
    .zero 8
    .globl glob_global_b
glob_global_b:
    .zero 8
    .text
    .globl main

gcd:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
.L0:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $0, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setne %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq -96(%rbp), %rax
    leave
    ret

lcm:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq gcd(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    movq -40(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -48(%rbp)
    movq -24(%rbp), %rax
    movq -48(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    leave
    ret

test_gcd_lcm:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    leaq gcd(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    leaq lcm(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    movq -56(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq glob_global_a(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_global_b(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq test_gcd_lcm(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $100, %rax
    movq %rax, -24(%rbp)
    movq $50, %rax
    movq %rax, -32(%rbp)
    leaq test_gcd_lcm(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $17, %rax
    movq %rax, -40(%rbp)
    movq $19, %rax
    movq %rax, -48(%rbp)
    leaq test_gcd_lcm(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -48(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $120, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rcx
    movq %rcx, glob_global_a(%rip)
    movq $75, %rax
    movq %rax, -64(%rbp)
    movq -64(%rbp), %rcx
    movq %rcx, glob_global_b(%rip)
    movq glob_global_a(%rip), %rcx
    movq %rcx, -72(%rbp)
    movq glob_global_b(%rip), %rcx
    movq %rcx, -80(%rbp)
    leaq test_gcd_lcm(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    movq -80(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
