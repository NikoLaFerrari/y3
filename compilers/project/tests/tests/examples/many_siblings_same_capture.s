    .data
    .text
    .globl main

main$add1:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
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
    leave
    ret

main$add2:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main$add3:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $3, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main$add4:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $4, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main$add5:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq 24(%rbp), %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $5, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq 24(%rbp), %rax
    movq -24(%rbp), %rcx
    movq %rcx, 0(%rax)
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $224, %rsp
    movq $0, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    leaq main$add1(%rip), %rax
    movq %rax, -160(%rbp)
    movq %rbp, -152(%rbp)
    leaq -160(%rbp), %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    leaq main$add2(%rip), %rax
    movq %rax, -176(%rbp)
    movq %rbp, -168(%rbp)
    leaq -176(%rbp), %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -24(%rax)
    leaq main$add3(%rip), %rax
    movq %rax, -192(%rbp)
    movq %rbp, -184(%rbp)
    leaq -192(%rbp), %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -32(%rax)
    leaq main$add4(%rip), %rax
    movq %rax, -208(%rbp)
    movq %rbp, -200(%rbp)
    leaq -208(%rbp), %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -40(%rax)
    leaq main$add5(%rip), %rax
    movq %rax, -224(%rbp)
    movq %rbp, -216(%rbp)
    leaq -224(%rbp), %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -104(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq -112(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq -120(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -128(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq -136(%rbp), %r11
    movq 0(%r11), %rax
    movq 8(%r11), %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -144(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
