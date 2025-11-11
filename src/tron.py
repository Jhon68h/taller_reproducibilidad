# tron.py
#   Interfaz esperada:
#   f_fun(w)              -> float
#   g_fun(w)              -> np.ndarray (gradiente)
#   hv_fun(w, s)          -> np.ndarray (H(w) @ s)
#
#
#   tron = Tron(f_fun, g_fun, hv_fun, eps=1e-2, max_iter=100)
#   w_opt, info = tron.tron(w0)

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple, Any

Array = np.ndarray


class Tron:
    """
    Trust-Region Newton con CG truncado (TRON).
    Traduce la lógica clásica de LIBLINEAR/TRON:
      - Región de confianza con actualización por ratio = actred / prered
      - Subproblema interno resuelto con CG truncado y proyección en la bola ||p|| <= delta
      - Criterios de parada por norma de gradiente relativa y tope de iteraciones
    """

    def __init__(
        self,
        f_fun: Callable[[Array], float],
        g_fun: Callable[[Array], Array],
        hv_fun: Callable[[Array, Array], Array],
        eps: float = 1e-2,
        max_iter: int = 100,
        verbose: bool = True,
        cg_tol_factor: float = 0.5,
        eta1: float = 0.25,
        eta2: float = 0.75,
        sigma1: float = 0.25,
        sigma2: float = 2.0,
        delta_init: float | None = None,
        delta_max: float | None = None,
        callback: Callable[[Dict[str, Any]], None] | None = None,
    ):
        """
        Parámetros:
          eps: tolerancia relativa sobre ||g|| (para ||g|| <= eps * ||g0||)
          max_iter: máximo de iteraciones externas TRON
          verbose: imprime trazas por iteración si True
          cg_tol_factor: controla la tolerancia interna del CG:
                         tol_k = min(0.5, sqrt(||g||)) * ||g|| * cg_tol_factor
          eta1, eta2, sigma1, sigma2: umbrales/factores de actualización de delta
          delta_init: si None, se usa ||g0||
          delta_max: si None, no se acota superiormente
          callback: función opcional que recibe un dict con info por iteración
        """
        self.f_fun = f_fun
        self.g_fun = g_fun
        self.hv_fun = hv_fun
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.cg_tol_factor = float(cg_tol_factor)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.delta_init = delta_init
        self.delta_max = delta_max
        self.callback = callback
        
    @staticmethod
    def _project_to_ball(p: Array, delta: float) -> Array:
        nrm = np.linalg.norm(p)
        if nrm <= delta:
            return p
        if nrm == 0.0:
            return p
        return p * (delta / nrm)


    def trcg(self,w: Array,g: Array,delta: float,tol: float,) -> Tuple[Array, int]:
        """
        Resuelve aproximadamente: (H) p ≈ -g con CG truncado,
        sujeto a ||p|| <= delta (proyección al cruzar el radio).
        Devuelve:
          p: dirección
          iters: iteraciones de CG realizadas
        """
        n = g.size
        p = np.zeros(n, dtype=np.float64)

        r = -g.copy()           # residual inicial: b - A p con b=-g y p=0
        d = r.copy()            # dirección de búsqueda
        rr = float(r @ r)
        if rr == 0.0:
            return p, 0

        for it in range(1, 10_000_000):  # tope práctico, salimos por tolerancia
            # z = H d
            z = self.hv_fun(w, d)
            dz = float(d @ z)

            if dz <= 0:
                # Matriz mal condicionada/no definida: dar un paso hasta el borde
                # alfa = argmin ||r - alpha * z||^2 limitado por ||p + alpha d|| <= delta
                # sin entrar en detalles, caemos al paso de proyección conservadora:
                alpha = 1.0
            else:
                alpha = rr / dz
            # Proponer nuevo p y proyectar si excede el radio
            p_new = p + alpha * d
            nrm_new = np.linalg.norm(p_new)
            if nrm_new > delta:
                # Cortar en la intersección con la esfera
                # Resolver para tau en ||p + tau d|| = delta
                # ||p||^2 + 2 tau p·d + tau^2 ||d||^2 = delta^2
                pd = float(p @ d)
                dd = float(d @ d)
                pp = float(p @ p)
                # tau = (-pd + sqrt(pd^2 + dd*(delta^2 - pp)))/dd  (raíz positiva)
                rad = pd * pd + dd * (delta * delta - pp)
                tau = 0.0 if dd == 0 else (-pd + np.sqrt(max(0.0, rad))) / dd
                p = p + tau * d
                return p, it  # truncación por borde

            # Aceptar paso CG completo
            p = p_new
            r = r - alpha * z
            rr_new = float(r @ r)

            # Criterio de parada interno
            if np.sqrt(rr_new) <= tol:
                return p, it

            beta = rr_new / rr
            d = r + beta * d
            rr = rr_new

        # Deberíamos haber salido antes
        return p, it # type: ignore


    def tron(self, w0: Array) -> Tuple[Array, Dict[str, Any]]:
        w = w0.astype(np.float64, copy=True)

        f = float(self.f_fun(w))
        g = self.g_fun(w).astype(np.float64, copy=False)
        g0_norm = float(np.linalg.norm(g))
        g_norm = g0_norm

        # región inicial
        delta = self.delta_init if self.delta_init is not None else max(1.0, g0_norm)
        if self.delta_max is not None:
            delta = min(delta, float(self.delta_max))

        # tolerancia CG por iteración (dependiente de ||g||)
        def cg_tol(gn: float) -> float:
            return min(0.5, np.sqrt(max(gn, 1e-32))) * gn * self.cg_tol_factor

        history = []
        if self.verbose:
            print(f"{'iter':>4} {'f':>14} {'||g||':>12} {'delta':>10} {'actred':>12} "
                  f"{'prered':>12} {'ratio':>9} {'CG':>5}")

        # Criterio de parada inmediato (g==0)
        if g0_norm == 0.0:
            info = {
                "iters": 0,
                "f": f,
                "g_norm": g_norm,
                "delta": delta,
                "history": history,
                "reason": "g0 == 0",
            }
            return w, info

        for k in range(1, self.max_iter + 1):
            # Subproblema: dirección p por CG truncado
            tol_k = cg_tol(g_norm)
            p, cg_iters = self.trcg(w, g, delta, tol=tol_k)

            if np.allclose(p, 0.0):
                # No progreso en subproblema
                info = {
                    "iters": k - 1,
                    "f": f,
                    "g_norm": g_norm,
                    "delta": delta,
                    "history": history,
                    "reason": "p ~ 0",
                }
                if self.verbose:
                    print("Dirección nula, salida.")
                return w, info

            # Predicted reduction: prered = - (g^T p + 0.5 p^T H p)
            Hp = self.hv_fun(w, p)
            prered = -float(g @ p + 0.5 * (p @ Hp))

            # Evaluar candidato
            w_new = w + p
            f_new = float(self.f_fun(w_new))
            actred = f - f_new
            ratio = actred / prered if prered > 0 else -np.inf

            # Actualizar región de confianza
            if ratio < self.eta1:
                delta *= self.sigma1
            else:
                if ratio > self.eta2 and np.isclose(np.linalg.norm(p), delta, rtol=1e-7, atol=1e-12):
                    delta = min(self.sigma2 * delta, self.delta_max or (self.sigma2 * delta))

            # Aceptar o rechazar paso
            accepted = ratio > 1e-4 and actred > 0.0
            if accepted:
                w = w_new
                f = f_new
                g = self.g_fun(w).astype(np.float64, copy=False)
                g_norm = float(np.linalg.norm(g))

            # Log/traza
            row = {
                "iter": k,
                "f": f,
                "g_norm": g_norm,
                "delta": delta,
                "actred": actred,
                "prered": prered,
                "ratio": ratio,
                "cg": cg_iters,
                "accepted": accepted,
            }
            history.append(row)
            if self.verbose:
                print(f"{k:4d} {f:14.6e} {g_norm:12.4e} {delta:10.4e} "
                      f"{actred:12.4e} {prered:12.4e} {ratio:9.4f} {cg_iters:5d}")

            if self.callback is not None:
                try:
                    self.callback(row)
                except Exception:
                    pass  # no abortar la optimización por el callback

            # Criterio de parada
            if g_norm <= self.eps * max(1.0, g0_norm):
                info = {
                    "iters": k,
                    "f": f,
                    "g_norm": g_norm,
                    "delta": delta,
                    "history": history,
                    "reason": "grad_tol",
                }
                return w, info

        info = {
            "iters": self.max_iter,
            "f": f,
            "g_norm": g_norm,
            "delta": delta,
            "history": history,
            "reason": "max_iter",
        }
        return w, info


# Resuelve una cuadrática convexa: f(w)=0.5 w^T A w - b^T w con A pd,
# para validar que converge hacia A^{-1} b.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 50
    U = rng.normal(size=(n, n))
    A = U.T @ U + 1e-3 * np.eye(n)
    b = rng.normal(size=n)

    def f_fun(w: Array) -> float:
        return 0.5 * float(w @ (A @ w)) - float(b @ w)

    def g_fun(w: Array) -> Array:
        return (A @ w) - b

    def hv_fun(w: Array, s: Array) -> Array:
        return A @ s

    w0 = np.zeros(n)
    tron = Tron(f_fun, g_fun, hv_fun, eps=1e-8, max_iter=100, verbose=True)
    w_opt, info = tron.tron(w0)
    # Solución cerrada
    w_star = np.linalg.solve(A, b)
    err = np.linalg.norm(w_opt - w_star) / max(1.0, np.linalg.norm(w_star))
    print(f"Relative error vs closed-form: {err:.3e}")
