    .data
    .text
    .globl main

evaluate_complex:
    pushq %rbp
    movq %rsp, %rbp
    subq $112, %rsp
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
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -72(%rbp)
    movq $2, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -88(%rbp)
    movq -48(%rbp), %rax
    movq -88(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -104(%rbp), %rax
    leave
    ret

nested_arithmetic:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $3, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq $2, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -40(%rbp)
    movq $5, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $1, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -80(%rbp)
    movq -56(%rbp), %rax
    movq -80(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $304, %rsp
    movq $10, %rax
    movq %rax, -32(%rbp)
    movq $6, %rax
    movq %rax, -40(%rbp)
    movq $3, %rax
    movq %rax, -48(%rbp)
    leaq evaluate_complex(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    movq -40(%rbp), %rsi
    movq -48(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $20, %rax
    movq %rax, -64(%rbp)
    movq $10, %rax
    movq %rax, -72(%rbp)
    movq $2, %rax
    movq %rax, -80(%rbp)
    leaq evaluate_complex(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    movq -72(%rbp), %rsi
    movq -80(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $5, %rax
    movq %rax, -96(%rbp)
    leaq nested_arithmetic(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -104(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -112(%rbp)
    leaq nested_arithmetic(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $7, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -128(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $3, %rax
    movq %rax, -136(%rbp)
    movq %rbp, %rax
    movq -136(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -160(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -184(%rbp)
    movq -160(%rbp), %rax
    movq -184(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -192(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -216(%rbp)
    movq -192(%rbp), %rax
    movq -216(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -224(%rbp)
    movq %rbp, %rax
    movq -224(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -232(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -232(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $15, %rax
    movq %rax, -240(%rbp)
    movq $5, %rax
    movq %rax, -248(%rbp)
    movq -240(%rbp), %rax
    movq -248(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -256(%rbp)
    movq $2, %rax
    movq %rax, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -272(%rbp)
    movq $10, %rax
    movq %rax, -280(%rbp)
    movq $2, %rax
    movq %rax, -288(%rbp)
    movq -280(%rbp), %rax
    movq -288(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -296(%rbp)
    movq -272(%rbp), %rax
    movq -296(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -304(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -304(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
