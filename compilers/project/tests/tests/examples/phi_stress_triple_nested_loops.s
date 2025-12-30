    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $688, %rsp
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $0, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $0, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq $0, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq $0, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -56(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq $4, %rax
    movq %rax, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -136(%rbp)
    movq -136(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq $0, %rax
    movq %rax, -144(%rbp)
    movq %rbp, %rax
    movq -144(%rbp), %rcx
    movq %rcx, -16(%rax)
.L3:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq $4, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    testq %rax, %rax
    jnz .L4
    jmp .L5
.L4:
    movq $0, %rax
    movq %rax, -176(%rbp)
    movq %rbp, %rax
    movq -176(%rbp), %rcx
    movq %rcx, -24(%rax)
.L6:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -184(%rbp)
    movq $4, %rax
    movq %rax, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -200(%rbp)
    movq -200(%rbp), %rax
    testq %rax, %rax
    jnz .L7
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -208(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq -208(%rbp), %rax
    movq -216(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -224(%rbp)
    movq -224(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -248(%rbp)
    movq -248(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq $1, %rax
    movq %rax, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -272(%rbp)
    movq %rbp, %rax
    movq -272(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -280(%rbp)
    movq $2, %rax
    movq %rax, -288(%rbp)
    movq -280(%rbp), %rax
    movq -288(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -296(%rbp)
    movq %rbp, %rax
    movq -296(%rbp), %rcx
    movq %rcx, -40(%rax)
    jmp .L14
.L13:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -304(%rbp)
    movq $3, %rax
    movq %rax, -312(%rbp)
    movq -304(%rbp), %rax
    movq -312(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -320(%rbp)
    movq %rbp, %rax
    movq -320(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -328(%rbp)
    movq $4, %rax
    movq %rax, -336(%rbp)
    movq -328(%rbp), %rax
    movq -336(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -344(%rbp)
    movq %rbp, %rax
    movq -344(%rbp), %rcx
    movq %rcx, -48(%rax)
.L14:
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -352(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -360(%rbp)
    movq -352(%rbp), %rax
    movq -360(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -368(%rbp)
    movq -368(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -376(%rbp)
    movq $5, %rax
    movq %rax, -384(%rbp)
    movq -376(%rbp), %rax
    movq -384(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -392(%rbp)
    movq %rbp, %rax
    movq -392(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -400(%rbp)
    movq $6, %rax
    movq %rax, -408(%rbp)
    movq -400(%rbp), %rax
    movq -408(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -416(%rbp)
    movq %rbp, %rax
    movq -416(%rbp), %rcx
    movq %rcx, -56(%rax)
    jmp .L17
.L16:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -424(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq -424(%rbp), %rax
    movq -432(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -440(%rbp)
    movq -440(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -448(%rbp)
    movq $7, %rax
    movq %rax, -456(%rbp)
    movq -448(%rbp), %rax
    movq -456(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -464(%rbp)
    movq %rbp, %rax
    movq -464(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -472(%rbp)
    movq $8, %rax
    movq %rax, -480(%rbp)
    movq -472(%rbp), %rax
    movq -480(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -488(%rbp)
    movq %rbp, %rax
    movq -488(%rbp), %rcx
    movq %rcx, -56(%rax)
    jmp .L20
.L19:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -496(%rbp)
    movq $9, %rax
    movq %rax, -504(%rbp)
    movq -496(%rbp), %rax
    movq -504(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -512(%rbp)
    movq %rbp, %rax
    movq -512(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -520(%rbp)
    movq $10, %rax
    movq %rax, -528(%rbp)
    movq -520(%rbp), %rax
    movq -528(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -536(%rbp)
    movq %rbp, %rax
    movq -536(%rbp), %rcx
    movq %rcx, -40(%rax)
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -544(%rbp)
    movq $11, %rax
    movq %rax, -552(%rbp)
    movq -544(%rbp), %rax
    movq -552(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -560(%rbp)
    movq %rbp, %rax
    movq -560(%rbp), %rcx
    movq %rcx, -48(%rax)
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -568(%rbp)
    movq $12, %rax
    movq %rax, -576(%rbp)
    movq -568(%rbp), %rax
    movq -576(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -584(%rbp)
    movq %rbp, %rax
    movq -584(%rbp), %rcx
    movq %rcx, -56(%rax)
.L20:
.L17:
.L11:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -592(%rbp)
    movq $1, %rax
    movq %rax, -600(%rbp)
    movq -592(%rbp), %rax
    movq -600(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -608(%rbp)
    movq %rbp, %rax
    movq -608(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L6
.L8:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -616(%rbp)
    movq $1, %rax
    movq %rax, -624(%rbp)
    movq -616(%rbp), %rax
    movq -624(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -632(%rbp)
    movq %rbp, %rax
    movq -632(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L3
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -640(%rbp)
    movq $1, %rax
    movq %rax, -648(%rbp)
    movq -640(%rbp), %rax
    movq -648(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -656(%rbp)
    movq %rbp, %rax
    movq -656(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -664(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -664(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -672(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -672(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -48(%rax), %rcx
    movq %rcx, -680(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -680(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -56(%rax), %rcx
    movq %rcx, -688(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -688(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
