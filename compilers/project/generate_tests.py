import os
import random

def generate_tests():
    output_dir = "tests_generated"
    os.makedirs(output_dir, exist_ok=True)

    test_count = 0

    def write_test(name, content):
        nonlocal test_count
        filename = f"{output_dir}/test_{test_count:03d}_{name}.bx"
        with open(filename, "w") as f:
            f.write(content)
        test_count += 1

    # --- 1. Basic Arithmetic & Logic (20 tests) ---
    for i in range(20):
        ops = ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>']
        op = random.choice(ops)
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        
        # Avoid division by zero
        if op in ['/', '%']:
            b = max(1, b)
            
        content = f"""
def main() {{
    var x = {a} : int;
    var y = {b} : int;
    var res = 0 : int;
    res = x {op} y;
    print(res);
}}
"""
        write_test(f"arith_{op_name(op)}", content)

    # --- 2. Control Flow (If/Else) (10 tests) ---
    for i in range(10):
        val = random.randint(0, 20)
        threshold = random.randint(0, 20)
        content = f"""
def main() {{
    var x = {val} : int;
    if (x < {threshold}) {{
        print(1);
    }} else {{
        print(0);
    }}
}}
"""
        write_test(f"if_else_{i}", content)

    # --- 3. Control Flow (While/Break/Continue) (10 tests) ---
    for i in range(10):
        limit = random.randint(5, 15)
        content = f"""
def main() {{
    var i = 0 : int;
    while (i < {limit}) {{
        i = i + 1;
        if (i % 2 == 0) {{
            continue;
        }}
        print(i);
        if (i > {limit - 2}) {{
            break;
        }}
    }}
}}
"""
        write_test(f"loop_{i}", content)

    # --- 4. Basic Functions (10 tests) ---
    for i in range(10):
        arg = random.randint(1, 10)
        content = f"""
def square(x: int): int {{
    return x * x;
}}

def main() {{
    print(square({arg}));
}}
"""
        write_test(f"func_basic_{i}", content)

    # --- 5. Recursion (Factorial) (5 tests) ---
    for i in range(5):
        n = random.randint(3, 8)
        content = f"""
def fact(n: int): int {{
    if (n <= 1) {{ return 1; }}
    return n * fact(n - 1);
}}

def main() {{
    print(fact({n}));
}}
"""
        write_test(f"recursion_fact_{i}", content)

    # --- 6. Recursion (Fibonacci) (5 tests) ---
    for i in range(5):
        n = random.randint(3, 10)
        content = f"""
def fib(n: int): int {{
    if (n <= 1) {{ return n; }}
    return fib(n - 1) + fib(n - 2);
}}

def main() {{
    print(fib({n}));
}}
"""
        write_test(f"recursion_fib_{i}", content)

    # --- 7. Nested Functions & Scoping (10 tests) ---
    for i in range(10):
        outer_val = random.randint(10, 20)
        inner_val = random.randint(1, 5)
        content = f"""
def main() {{
    var x = {outer_val} : int;
    
    def inner(y: int): int {{
        return x + y;
    }}
    
    print(inner({inner_val}));
}}
"""
        write_test(f"nested_scope_{i}", content)

    # --- 8. Variable Capture & Mutation (10 tests) ---
    for i in range(10):
        start_val = random.randint(0, 5)
        content = f"""
def main() {{
    var count = {start_val} : int;
    
    def increment(): void {{
        count = count + 1;
    }}
    
    increment();
    increment();
    print(count);
}}
"""
        write_test(f"capture_mutation_{i}", content)

    # --- 9. Higher-Order Functions (10 tests) ---
    for i in range(10):
        val = random.randint(1, 10)
        content = f"""
def apply(f: function(int) -> int, x: int): int {{
    return f(x);
}}

def double(n: int): int {{
    return n * 2;
}}

def main() {{
    print(apply(double, {val}));
}}
"""
        write_test(f"higher_order_{i}", content)

    # --- 10. Complex Higher-Order (Closure passing) (10 tests) ---
    for i in range(10):
        factor = random.randint(2, 5)
        arg = random.randint(1, 10)
        content = f"""
def apply_func(f: function(int) -> int, arg: int): int {{
    return f(arg);
}}

def main() {{
    var multiplier = {factor} : int;
    
    def mult(x: int): int {{
        return x * multiplier;
    }}
    
    print(apply_func(mult, {arg}));
}}
"""
        write_test(f"closure_passing_{i}", content)

    print(f"Generated {test_count} tests in '{output_dir}'")

def op_name(op):
    names = {
        '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod',
        '&': 'and', '|': 'or', '^': 'xor', '<<': 'shl', '>>': 'shr'
    }
    return names.get(op, 'op')

if __name__ == "__main__":
    generate_tests()
