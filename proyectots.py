import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = {
    "M": "^GSPC",   
    "A": "MSFT",   
    "B": "MO",     
    "C": "NEM"    
}

inicio = "2021-10-01"
fin = "2026-10-31"


raw_data = yf.download(
    list(tickers.values()),
    start=inicio,
    end=fin,
    interval="1mo",
    auto_adjust=True,
    progress=False
)["Close"]
inv_map = {v: k for k, v in tickers.items()}
precios = raw_data.rename(columns=inv_map)[["M", "A", "B", "C"]].dropna()


rendimientos = precios.pct_change().dropna()
N = len(rendimientos)
pi_s = 1.0 / N  

assert N == 60, f"Atención: se obtuvieron {N} rendimientos en lugar de 60. Ajusta las fechas."


rf_raw = yf.download("^IRX", start=inicio, end=fin, interval="1mo", progress=False)["Close"].dropna()
rf_raw = rf_raw.loc[rendimientos.index] 
rf_anual_prom = float(rf_raw.mean().iloc[0]) / 100.0  
r_f = (1.0 + rf_anual_prom) ** (1.0 / 12.0) - 1.0    


precios.to_csv("tabla_precios_mensuales.csv")
rendimientos.to_csv("tabla_rendimientos_mensuales.csv")

print(f"Observaciones (N): {N} | Probabilidad por estado (pi_s): {pi_s:.6f}")
print(f"Tasa libre de riesgo anual promedio: {rf_anual_prom:.6%} | r_f mensual: {r_f:.6%}\n")

activos = ["M", "A", "B", "C"]
r = {}      
var = {}   
std = {}    
p_riesgo = {} 

print("=== PUNTOS 1 A 4: ESTADÍSTICOS INDIVIDUALES Y PRECIO DEL RIESGO ===")
for i in activos:
    r_is = rendimientos[i].to_numpy(dtype=np.float64)
    r[i] = np.sum(pi_s * r_is)
    var[i] = np.sum(pi_s * (r_is - r[i]) ** 2)
    std[i] = np.sqrt(var[i])
    p_riesgo[i] = (r[i] - r_f) / std[i]
    print(f"Activo {i} ({tickers[i]}): r_{i} = {r[i]:.6%} | sigma_{i}^2 = {var[i]:.8f} | "
          f"sigma_{i} = {std[i]:.6%} | p_{i} = {p_riesgo[i]:.6f}")

mejor_p = max(p_riesgo, key=p_riesgo.get)
print(f"-> Mayor precio del riesgo: {mejor_p} ({p_riesgo[mejor_p]:.6f})\n")

sigma_x = std["M"]
x_inv = {}
r_x = {}

print(f"=== PUNTO 5: CONJUNTO DE OPORTUNIDADES CON RIESGO OBJETIVO sigma_x = {sigma_x:.6%} ===")
for i in activos:
    x_inv[i] = sigma_x / std[i]
    r_x[i] = x_inv[i] * r[i] + (1.0 - x_inv[i]) * r_f
    print(f"Combinación F-{i}: x_{i} = {x_inv[i]:.6f} | (1 - x_{i}) = {1 - x_inv[i]:.6f} | r_x = {r_x[i]:.6%}")

mejor_oportunidad = max(r_x, key=r_x.get)
print(f"-> Alternativa que más amplía el conjunto de oportunidades: F-{mejor_oportunidad} (r_x = {r_x[mejor_oportunidad]:.6%})\n")

pares = [("A", "B"), ("A", "C"), ("B", "C")]
cov_pares = {}
rho_pares = {}

print("=== PUNTO 6: COVARIANZAS Y CORRELACIONES ENTRE ACTIVOS ===")
for i, j in pares:
    r_is = rendimientos[i].to_numpy(dtype=np.float64)
    r_js = rendimientos[j].to_numpy(dtype=np.float64)
    cov_pares[(i, j)] = np.sum(pi_s * (r_is - r[i]) * (r_js - r[j]))
    rho_pares[(i, j)] = cov_pares[(i, j)] / (std[i] * std[j])
    print(f"Par {i}-{j}: Cov(R_{i}, R_{j}) = {cov_pares[(i, j)]:.8f} | rho_{{{i},{j}}} = {rho_pares[(i, j)]:.6f}")
print()

print("=== PUNTO 7: CARTERA P (A, B, C) CON PESOS PROPORCIONALES A p_i ===")
if all(p_riesgo[k] > 0 for k in ["A", "B", "C"]):
    suma_p = p_riesgo["A"] + p_riesgo["B"] + p_riesgo["C"]
    w = {k: p_riesgo[k] / suma_p for k in ["A", "B", "C"]}
else:
    raise ValueError("Al menos un activo tiene p_i <= 0 en esta ventana; conviene asignar pesos manuales positivos.")

assert np.isclose(sum(w.values()), 1.0), "Error: los pesos no suman 1."

r_P_s = (w["A"] * rendimientos["A"].to_numpy(dtype=np.float64) +
         w["B"] * rendimientos["B"].to_numpy(dtype=np.float64) +
         w["C"] * rendimientos["C"].to_numpy(dtype=np.float64))


rendimientos["r_P"] = r_P_s
rendimientos.to_csv("tabla_rendimientos_con_cartera_P.csv")

r_P = np.sum(pi_s * r_P_s)
var_P = np.sum(pi_s * (r_P_s - r_P) ** 2)
std_P = np.sqrt(var_P)
std_ponderada_sin_div = w["A"] * std["A"] + w["B"] * std["B"] + w["C"] * std["C"]

print(f"Pesos: w_A = {w['A']:.6f}, w_B = {w['B']:.6f}, w_C = {w['C']:.6f} (Suma = {sum(w.values()):.6f})")
print(f"Cartera P: r_P = {r_P:.6%} | sigma_P^2 = {var_P:.8f} | sigma_P = {std_P:.6%}")
print(f"Promedio ponderado de desviaciones individuales: {std_ponderada_sin_div:.6%}")
print(f"Reducción de riesgo por diversificación: {(std_ponderada_sin_div - std_P):.6%}\n")

print("=== PUNTOS 8 A 13: RIESGO SISTEMÁTICO, BETAS Y EQUILIBRIO CAPM ===")
m_s = rendimientos["M"].to_numpy(dtype=np.float64)
r_m = r["M"]
var_m = var["M"]
std_m = std["M"]

cov_m = {}
beta = {}
riesgo_mercado = {}
r_ajustado = {}
r_capm = {}

for i in ["A", "B", "C"]:
    r_is = rendimientos[i].to_numpy(dtype=np.float64)

    cov_m[i] = np.sum(pi_s * (r_is - r[i]) * (m_s - r_m))

    beta[i] = cov_m[i] / var_m

    riesgo_mercado[i] = beta[i] * std_m

    r_ajustado[i] = r[i] - beta[i] * (r_m - r_f)
   
    r_capm[i] = r_f + beta[i] * (r_m - r_f)
    
    diferencia = r[i] - r_capm[i]
    if np.isclose(diferencia, 0.0, atol=1e-7):
        posicion = "SOBRE la recta de mercado (en equilibrio)"
    elif diferencia > 0:
        posicion = "POR ENCIMA de la recta de mercado (infravalorado)"
    else:
        posicion = "POR DEBAJO de la recta de mercado (sobrevalorado)"

    print(f"--- Activo {i} ({tickers[i]}) ---")
    print(f"  8. Cov(R_{i}, R_m)      = {cov_m[i]:.8f}")
    print(f"  9. beta_{i}             = {beta[i]:.6f}")
    print(f" 10. Riesgo de mercado    = {riesgo_mercado[i]:.6%}")
    print(f" 11. Rend. ajustado obs.  = {r_ajustado[i]:.6%} (vs r_f = {r_f:.6%})")
    print(f" 12. CAPM (r_{i}*)        = {r_capm[i]:.6%}")
    print(f" 13. Clasificación        = {posicion} (Alfa = {diferencia:+.6%})\n")

betas_grid = np.linspace(0, max(max(beta.values()), 1.2) * 1.15, 100)
sml_grid = r_f + betas_grid * (r_m - r_f)

plt.figure(figsize=(8, 5))
plt.plot(betas_grid, sml_grid * 100, label="Recta del Mercado de Valores (CAPM)", color="navy", linewidth=2)
plt.scatter([1.0], [r_m * 100], color="black", zorder=5, label="Mercado M (S&P 500)")

colores = {"A": "teal", "B": "darkorange", "C": "crimson"}
for i in ["A", "B", "C"]:
    plt.scatter([beta[i]], [r[i] * 100], color=colores[i], s=70, zorder=6, label=f"{i} ({tickers[i]})")
    plt.annotate(f" {i} ({tickers[i]})", (beta[i], r[i] * 100), fontsize=9)

plt.title("Recta del Mercado de Valores (CAPM) - Rendimiento Mensual Esperado vs. Beta")
plt.xlabel("Beta ($\\beta_i$)")
plt.ylabel("Rendimiento mensual esperado (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("grafica_CAPM_SML.png", dpi=300)
plt.show()
