.data
.text
.globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $48, %rsp
.LLentry:
  movq $42, -8(%rbp)
  movq -8(%rbp), %r11
  movq %r11, -16(%rbp)
  movq $0, -24(%rbp)
  movq -24(%rbp), %r11
  movq %r11, -32(%rbp)
  movq $0, -40(%rbp)
  movq -40(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %rdi
  callq bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
