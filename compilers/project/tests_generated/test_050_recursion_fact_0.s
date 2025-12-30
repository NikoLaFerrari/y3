    .text
    .globl main

fact:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq %r10, -8(%rbp)
    movq %rdi, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq $1, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $1, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
.L2:
    movq -16(%rbp), %rcx
    movq %rcx, -56(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -64(%rbp)
    movq $1, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -80(%rbp)
    leaq fact(%rip), %rax
    movq $0, %r10
    movq -80(%rbp), %rdi
    call *%rax
    movq %rax, -88(%rbp)
    movq -56(%rbp), %rax
    movq -88(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -96(%rbp)
    movq -96(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %r10, -8(%rbp)
    movq $4, %rax
    movq %rax, -16(%rbp)
    leaq fact(%rip), %rax
    movq $0, %r10
    movq -16(%rbp), %rdi
    call *%rax
    movq %rax, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -24(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret
