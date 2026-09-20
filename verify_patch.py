import ast
import sys

files = [
    "relational_attention/attention.py",
    "relational_attention/layers.py",
    "relational_attention/model.py",
]

ok = True
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f, "OK")
    except SyntaxError as e:
        print(f, "SYNTAX ERROR:", e)
        ok = False

sys.exit(0 if ok else 1)
