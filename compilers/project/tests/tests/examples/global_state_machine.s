    .data
    .globl glob_state
glob_state:
    .zero 8
    .globl glob_value
glob_value:
    .zero 8
    .text
    .globl main

increment_state:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_state(%rip)
    leave
    ret

process:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq glob_value(%rip), %rcx
    movq %rcx, -32(%rbp)
    movq $10, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rcx
    movq %rcx, glob_value(%rip)
    jmp .L2
.L1:
    movq glob_state(%rip), %rcx
    movq %rcx, -56(%rbp)
    movq $1, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq glob_value(%rip), %rcx
    movq %rcx, -80(%rbp)
    movq $2, %rax
    movq %rax, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -96(%rbp)
    movq -96(%rbp), %rcx
    movq %rcx, glob_value(%rip)
    jmp .L5
.L4:
    movq glob_state(%rip), %rcx
    movq %rcx, -104(%rbp)
    movq $2, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq glob_value(%rip), %rcx
    movq %rcx, -128(%rbp)
    movq $5, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -144(%rbp)
    movq -144(%rbp), %rcx
    movq %rcx, glob_value(%rip)
    jmp .L8
.L7:
    movq $0, %rax
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rcx
    movq %rcx, glob_value(%rip)
.L8:
.L5:
.L2:
    leave
    ret

step:
    pushq %rbp
    movq %rsp, %rbp
    subq $0, %rsp
    leaq process(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq increment_state(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_value(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_value(%rip), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_value(%rip), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_value(%rip), %rcx
    movq %rcx, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq step(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_state(%rip), %rcx
    movq %rcx, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_value(%rip), %rcx
    movq %rcx, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
