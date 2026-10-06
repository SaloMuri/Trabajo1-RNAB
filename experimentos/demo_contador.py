"""Demo minima: descenso por gradiente simple con conteo de evaluaciones.
Uso: python -m experimentos.demo_contador --funcion rosenbrock --n 2 --lr 0.001 --iters 5000 --seed 42
"""
import argparse
import numpy as np
from src.funciones import FUNCIONES
from src.contador import Problema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcion", default="rosenbrock", choices=list(FUNCIONES))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = FUNCIONES[a.funcion]
    p = Problema(d["f"], d["grad"], n=a.n)
    rng = np.random.default_rng(a.seed)
    x = rng.uniform(*d["dominio"], size=a.n)
    print(f"x0 = {x}, f(x0) = {p.f(x):.6f}")
    for _ in range(a.iters):
        x = x - a.lr * p.grad(x)
    print(f"x final = {x}, f = {p.f(x):.6f}")
    print(f"n_f = {p.n_f}, n_grad = {p.n_grad}, total equivalente (en f) = {p.total_equivalente}")


if __name__ == "__main__":
    main()
