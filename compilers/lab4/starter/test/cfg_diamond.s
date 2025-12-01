.data
.text
.globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $64, %rsp
.LLentry:
  movq $10, -8(%rbp)
  movq -8(%rbp), %r11
  movq %r11, -16(%rbp)
  movq $5, -24(%rbp)
  movq -16(%rbp), %r11
  subq -24(%rbp), %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  cmpq $0, %r11
  jg .LL0
  jmp .LL1
.LL0:
  movq $1, -40(%rbp)
  movq -16(%rbp), %r11
  addq -40(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  movq %r11, -16(%rbp)
  jmp .LL2
.LL1:
  movq $1, -56(%rbp)
  movq -16(%rbp), %r11
  subq -56(%rbp), %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  movq %r11, -16(%rbp)
.LL2:
  movq -16(%rbp), %rdi
  callq bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
