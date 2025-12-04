    .text
    .globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $56, %rsp
.LB0:
  movq $10, -8(%rbp)
  movq $5, -24(%rbp)
  movq -8(%rbp), %r11
  cmpq -24(%rbp), %r11
  setg %r10b
  movzbq %r10b, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  cmpq $0, %r11
  jne .LL0
  jmp .LL1
.LL0:
  movq $1, -32(%rbp)
  movq -8(%rbp), %r11
  addq -32(%rbp), %r11
  movq %r11, -40(%rbp)
  movq -40(%rbp), %r11
  movq %r11, -8(%rbp)
  jmp .LL2
.LL1:
  movq $1, -48(%rbp)
  movq -8(%rbp), %r11
  subq -48(%rbp), %r11
  movq %r11, -56(%rbp)
  movq -56(%rbp), %r11
  movq %r11, -8(%rbp)
  jmp .LL2
.LL2:
  movq -8(%rbp), %rdi
  call _bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
