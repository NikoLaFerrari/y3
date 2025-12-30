    .text
    .globl main

fib:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
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
    movq -16(%rbp), %rcx
    movq %rcx, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
.L2:
    movq -16(%rbp), %rcx
    movq %rcx, -56(%rbp)
    movq $1, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -72(%rbp)
    leaq fib(%rip), %rax
    movq $0, %r10
    movq -72(%rbp), %rdi
    call *%rax
    movq %rax, -80(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -88(%rbp)
    movq $2, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -104(%rbp)
    leaq fib(%rip), %rax
    movq $0, %r10
    movq -104(%rbp), %rdi
    call *%rax
    movq %rax, -112(%rbp)
    movq -80(%rbp), %rax
    movq -112(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %r10, -8(%rbp)
    movq $6, %rax
    movq %rax, -16(%rbp)
    leaq fib(%rip), %rax
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
