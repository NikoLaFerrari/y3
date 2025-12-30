    .data
    .text
    .globl main

main$f1:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $2, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -48(%rbp)
    movq 24(%rbp), %rax
    movq -48(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main$f2:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $3, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -48(%rbp)
    movq 24(%rbp), %rax
    movq -48(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq $1, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $2, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -16(%rax)
    leaq main$f1(%rip), %rax
    movq %rax, -128(%rbp)
    movq %rbp, -120(%rbp)
    leaq -128(%rbp), %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -24(%rax)
    leaq main$f2(%rip), %rax
    movq %rax, -144(%rbp)
    movq %rbp, -136(%rbp)
    leaq -144(%rbp), %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -72(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq -96(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
