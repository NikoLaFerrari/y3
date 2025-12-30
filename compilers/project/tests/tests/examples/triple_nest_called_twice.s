    .data
    .text
    .globl main

main$nest1$nest2$nest3:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main$nest1$nest2:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    leaq main$nest1$nest2$nest3(%rip), %rax
    movq %rax, -40(%rbp)
    movq %rbp, -32(%rbp)
    leaq -40(%rbp), %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -24(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main$nest1:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    leaq main$nest1$nest2(%rip), %rax
    movq %rax, -40(%rbp)
    movq %rbp, -32(%rbp)
    leaq -40(%rbp), %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq -24(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
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
    movq $3, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -24(%rax)
    leaq main$nest1(%rip), %rax
    movq %rax, -120(%rbp)
    movq %rbp, -112(%rbp)
    leaq -120(%rbp), %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -72(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $20, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $30, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -104(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
