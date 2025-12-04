    .text
    .globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $32, %rsp
.LB0:
  movq $10, -8(%rbp)
  movq $1, -16(%rbp)
  movq -16(%rbp), %r11
  cmpq $0, %r11
  jne .LL0
  jmp .LL2
.LL0:
  movq $1, -24(%rbp)
  movq -24(%rbp), %rdi
  call _bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
.LL2:
  movq $0, -32(%rbp)
  movq -32(%rbp), %rdi
  call _bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
