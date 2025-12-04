    .text
    .globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $24, %rsp
.LB0:
  movq $42, -8(%rbp)
  movq $0, -16(%rbp)
  movq $0, -24(%rbp)
  movq -8(%rbp), %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -24(%rbp), %rdi
  call _bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
