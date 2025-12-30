    .data
    .text
    .globl main

print_sum:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -16(%rbp), %rax
    movq -24(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

print_range:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -16(%rbp), %rax
    movq -24(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq $10, %rax
    movq %rax, -8(%rbp)
    movq $20, %rax
    movq %rax, -16(%rbp)
    leaq print_sum(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $5, %rax
    movq %rax, -24(%rbp)
    movq $15, %rax
    movq %rax, -32(%rbp)
    leaq print_sum(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $1, %rax
    movq %rax, -40(%rbp)
    movq $5, %rax
    movq %rax, -48(%rbp)
    leaq print_range(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -48(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
