    .data
    .text
    .globl main

main$level1$level2$level3$level4:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq $1, %rax
    movq %rax, -8(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq -8(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq -16(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $3, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq $4, %rax
    movq %rax, -32(%rbp)
    movq 24(%rbp), %rax
    movq -32(%rbp), %rcx
    movq %rcx, 0(%rax)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -56(%rbp)
    movq 24(%rbp), %rax
    movq 24(%rax), %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -72(%rbp)
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leave
    ret

main$level1$level2$level3:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$level1$level2$level3$level4(%rip), %rax
    movq %rax, -56(%rbp)
    movq %rbp, -48(%rbp)
    leaq -56(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main$level1$level2:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$level1$level2$level3(%rip), %rax
    movq %rax, -56(%rbp)
    movq %rbp, -48(%rbp)
    leaq -56(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main$level1:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$level1$level2(%rip), %rax
    movq %rax, -56(%rbp)
    movq %rbp, -48(%rbp)
    leaq -56(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
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
    subq $64, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$level1(%rip), %rax
    movq %rax, -64(%rbp)
    movq %rbp, -56(%rbp)
    leaq -64(%rbp), %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
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
    movq $0, %rax
    leave
    ret
