    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $544, %rsp
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -136(%rbp)
    movq %rbp, %rax
    movq -136(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq %rbp, %rax
    movq -144(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -184(%rbp)
    movq %rbp, %rax
    movq -184(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq %rbp, %rax
    movq -192(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq %rbp, %rax
    movq -200(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -208(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -208(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -232(%rbp)
    movq %rbp, %rax
    movq -232(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq %rbp, %rax
    movq -240(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -248(%rbp)
    movq %rbp, %rax
    movq -248(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -256(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -256(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -272(%rbp)
    movq -264(%rbp), %rax
    movq -272(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -280(%rbp)
    movq %rbp, %rax
    movq -280(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq %rbp, %rax
    movq -288(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -296(%rbp)
    movq %rbp, %rax
    movq -296(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -304(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -304(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -320(%rbp)
    movq -312(%rbp), %rax
    movq -320(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -328(%rbp)
    movq %rbp, %rax
    movq -328(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -336(%rbp)
    movq %rbp, %rax
    movq -336(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -344(%rbp)
    movq %rbp, %rax
    movq -344(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -352(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -352(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -360(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -368(%rbp)
    movq -360(%rbp), %rax
    movq -368(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -376(%rbp)
    movq %rbp, %rax
    movq -376(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -384(%rbp)
    movq %rbp, %rax
    movq -384(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -392(%rbp)
    movq %rbp, %rax
    movq -392(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -400(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -400(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -408(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -416(%rbp)
    movq -408(%rbp), %rax
    movq -416(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -424(%rbp)
    movq %rbp, %rax
    movq -424(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq %rbp, %rax
    movq -432(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -440(%rbp)
    movq %rbp, %rax
    movq -440(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -448(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -448(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -456(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -464(%rbp)
    movq -456(%rbp), %rax
    movq -464(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -472(%rbp)
    movq %rbp, %rax
    movq -472(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -480(%rbp)
    movq %rbp, %rax
    movq -480(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -488(%rbp)
    movq %rbp, %rax
    movq -488(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -496(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -496(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -504(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -512(%rbp)
    movq -504(%rbp), %rax
    movq -512(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -520(%rbp)
    movq %rbp, %rax
    movq -520(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -528(%rbp)
    movq %rbp, %rax
    movq -528(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -536(%rbp)
    movq %rbp, %rax
    movq -536(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -544(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -544(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
