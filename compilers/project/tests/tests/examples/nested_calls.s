    .data
    .text
    .globl main

add:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

multiply:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
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
    movq -24(%rbp), %rax
    leave
    ret

compute:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    leaq add(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    leaq multiply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq $3, %rax
    movq %rax, -16(%rbp)
    movq $4, %rax
    movq %rax, -24(%rbp)
    movq $5, %rax
    movq %rax, -32(%rbp)
    leaq compute(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    movq -24(%rbp), %rsi
    movq -32(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -56(%rbp)
    movq $20, %rax
    movq %rax, -64(%rbp)
    movq $2, %rax
    movq %rax, -72(%rbp)
    leaq compute(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    movq -64(%rbp), %rsi
    movq -72(%rbp), %rdx
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
    movq $2, %rax
    movq %rax, -88(%rbp)
    movq $3, %rax
    movq %rax, -96(%rbp)
    leaq multiply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    movq -96(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -104(%rbp)
    movq $4, %rax
    movq %rax, -112(%rbp)
    movq $5, %rax
    movq %rax, -120(%rbp)
    leaq multiply(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    movq -120(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -128(%rbp)
    leaq add(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    movq -128(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -136(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -136(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
