    .data
    .text
    .globl main

apply:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main$add_y:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $112, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$add_y(%rip), %rax
    movq %rax, -104(%rbp)
    movq %rbp, -96(%rbp)
    leaq -104(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq $5, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $15, %rax
    movq %rax, -72(%rbp)
    leaq apply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    movq -72(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L2
.L1:
    movq $0, %rax
    movq %rax, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
.L2:
    movq $0, %rax
    leave
    ret
