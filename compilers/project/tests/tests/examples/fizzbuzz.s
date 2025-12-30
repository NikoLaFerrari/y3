    .data
    .text
    .globl main

fizzbuzz:
    pushq %rbp
    movq %rsp, %rbp
    subq $208, %rsp
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
    movq $3, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -56(%rbp)
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq $5, %rax
    movq %rax, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -96(%rbp)
    movq $0, %rax
    movq %rax, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -112(%rbp)
    movq -112(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $151515, %rax
    movq %rax, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L8
.L7:
    movq $333, %rax
    movq %rax, -128(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -128(%rbp), %rdi
    call *%rax
    addq $16, %rsp
.L8:
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq $5, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -152(%rbp)
    movq $0, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $555, %rax
    movq %rax, -176(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -176(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -184(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -184(%rbp), %rdi
    call *%rax
    addq $16, %rsp
.L11:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq $1, %rax
    movq %rax, -200(%rbp)
    movq -192(%rbp), %rax
    movq -200(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -208(%rbp)
    movq %rbp, %rax
    movq -208(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $0, %rax
    movq %rax, -8(%rbp)
    movq $100, %rax
    movq %rax, -16(%rbp)
    leaq fizzbuzz(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
