import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Proyecto final - Teoría del Seguro
# Opción 3: Inversión en tres activos inciertos más el mercado
# Integrantes:
#
# M = S&P 500 (mercado), A = Microsoft, B = Altria, C = Newmont
# F = T-Bills de EUA a 13 semanas (^IRX)
# Datos: precios de cierre mensuales ajustados, Yahoo Finance

tickers = {"M": "^GSPC", "A": "MSFT", "B": "MO", "C": "NEM"}
activos = ["M", "A", "B", "C"]
riesgosos = ["A", "B", "C"]
inicio = "2021-10-01"
fin = "2026-10-31"

# True  -> descarga los datos de Yahoo Finance y los guarda en csv
# False -> usa los csv ya guardados (así salen los mismos números que en el documento)
DESCARGAR = True

if DESCARGAR:
    import yfinance as yf

    raw = yf.download(list(tickers.values()), start=inicio, end=fin, interval="1mo", auto_adjust=True, progress=False)["Close"]
    inv_tickers = {v: k for k, v in tickers.items()}
    precios = raw.rename(columns=inv_tickers)[activos].dropna()

    # ^IRX viene como tasa anual en %
    irx = yf.download("^IRX", start=inicio, end=fin, interval="1mo", auto_adjust=True, progress=False)["Close"].squeeze().dropna()

    precios.to_csv("tabla_precios_mensuales.csv")
    irx.to_csv("tasa_libre_riesgo_IRX.csv")
    print("Datos descargados de Yahoo Finance el", pd.Timestamp.today().strftime("%d/%m/%Y"))
else:
    precios = pd.read_csv("tabla_precios_mensuales.csv", index_col=0, parse_dates=True)
    irx = pd.read_csv("tasa_libre_riesgo_IRX.csv", index_col=0, parse_dates=True).squeeze("columns")

# Rendimiento mensual de cada activo: r_s = P_s / P_(s-1) - 1
rend = precios.pct_change().dropna()
N = len(rend)
meses = rend.index.strftime("%Y-%m").rename("Mes")

# Cada mes se toma como un estado s y todos los meses tienen la misma probabilidad
pi_s = 1 / N

tabla_rend = rend.copy()
tabla_rend.index = meses
tabla_rend.insert(0, "pi_s", pi_s)
tabla_rend.to_csv("tabla_rendimientos_mensuales.csv")

print(f"Periodo: {meses[0]} a {meses[-1]} ({N} meses)")
print(f"π_s = 1/{N} = {pi_s:.6f} en todos los estados  |  Σ π_s = {tabla_rend['pi_s'].sum():.6f}")
print("Nota: los números se muestran redondeados, pero los cálculos usan todos los decimales\n")

# Tasa libre de riesgo: promedio de ^IRX en los mismos meses, convertida a tasa mensual equivalente
irx_periodo = irx.reindex(rend.index)
rf_anual = irx_periodo.mean() / 100
rf = (1 + rf_anual) ** (1 / 12) - 1

print("--- Tasa libre de riesgo (F) ---")
print(f"Promedio de ^IRX ({irx_periodo.count()} de {N} meses): {rf_anual:.4%} anual")
print(f"r_f = (1 + r_anual)^(1/12) - 1 = (1 + {rf_anual:.6f})^(1/12) - 1 = {rf:.6f} ({rf:.4%} mensual)\n")


def terminos(serie):
    # Son 60 términos y no caben todos, así que se muestran los primeros 3.
    # La tabla completa queda en tabla_operaciones_por_estado.csv
    texto = " + ".join(f"({v:.6f})" for v in serie.iloc[:3])
    return f"[{texto} + ...]"


# 1. Rendimiento esperado
r = (pi_s * rend).sum()

print("--- 1. Rendimiento esperado (mensual) ---")
print("r_i = Σ π_s r_i,s")
for i in activos:
    print(f"r_{i} = (1/{N}) × {terminos(rend[i])} = (1/{N}) × {rend[i].sum():.6f} = {r[i]:.6f} ({r[i]:.4%})")
print()

# 2. Varianza
desv = rend - r  # r_i,s - r_i
var = (pi_s * desv ** 2).sum()

print("--- 2. Varianza ---")
print("σ²_i = Σ π_s (r_i,s - r_i)²")
for i in activos:
    print(f"σ²_{i} = (1/{N}) × {terminos(desv[i] ** 2)} = (1/{N}) × {(desv[i] ** 2).sum():.6f} = {var[i]:.6f}")
print()

# 3. Desviación típica
std = np.sqrt(var)

print("--- 3. Desviación típica ---")
print("σ_i = √σ²_i")
for i in activos:
    print(f"σ_{i} = √{var[i]:.6f} = {std[i]:.6f} ({std[i]:.4%})")
print()

# 4. Precio del riesgo
p_riesgo = (r - rf) / std

print("--- 4. Precio del riesgo ---")
print("p_i = (r_i - r_f) / σ_i")
for i in activos:
    print(f"p_{i} = ({r[i]:.6f} - {rf:.6f}) / {std[i]:.6f} = {p_riesgo[i]:.4f}")
mejor_p = p_riesgo.idxmax()
print(f"Mayor precio del riesgo: {mejor_p} ({tickers[mejor_p]}) con p = {p_riesgo[mejor_p]:.4f}\n")

# 5. Carteras F-M, F-A, F-B y F-C con el mismo riesgo
# Riesgo objetivo: el del mercado (σ_x = σ_M)
sigma_x = std["M"]
x_inv = sigma_x / std
r_x = x_inv * r + (1 - x_inv) * rf

print(f"--- 5. Carteras con F al mismo riesgo (σ_x = σ_M = {sigma_x:.6f}) ---")
print("x_i = σ_x / σ_i  ;  r_x = x_i r_i + (1 - x_i) r_f")
for i in activos:
    print(f"F-{i}: x = {sigma_x:.6f} / {std[i]:.6f} = {x_inv[i]:.4f}  ;  "
          f"r_x = {x_inv[i]:.4f} × {r[i]:.6f} + ({1 - x_inv[i]:.4f}) × {rf:.6f} = {r_x[i]:.6f} ({r_x[i]:.4%})")
    if x_inv[i] > 1:
        print(f"      x > 1: se pide prestado a la tasa r_f para invertir más del 100% en {i}")
mejor_op = r_x.idxmax()
print(f"La que más amplía el conjunto de oportunidades: F-{mejor_op} (r_x = {r_x[mejor_op]:.4%})\n")

# 6. Covarianzas y correlaciones entre A, B y C
pares = [("A", "B"), ("A", "C"), ("B", "C")]
covs = {}
rho = {}

print("--- 6. Covarianzas y correlaciones ---")
print("Cov(R_i,R_j) = Σ π_s (r_i,s - r_i)(r_j,s - r_j)  ;  ρ_i,j = Cov(R_i,R_j) / (σ_i σ_j)")
for i, j in pares:
    prod = desv[i] * desv[j]
    c = (pi_s * prod).sum()
    covs[i + "-" + j] = c
    rho[i + "-" + j] = c / (std[i] * std[j])
    print(f"Cov(R_{i},R_{j}) = (1/{N}) × {terminos(prod)} = (1/{N}) × {prod.sum():.6f} = {c:.6f}")
    print(f"ρ_{i},{j} = {c:.6f} / ({std[i]:.6f} × {std[j]:.6f}) = {rho[i + '-' + j]:.4f}")
par_div = min(rho, key=rho.get)
print(f"Par con menor correlación (mayor potencial de diversificación): {par_div} (ρ = {rho[par_div]:.4f})\n")

# 7. Cartera P con A, B y C
# Pesos proporcionales al precio del riesgo: w_i = p_i / (p_A + p_B + p_C)
print("--- 7. Cartera P (A, B y C) ---")
if (p_riesgo[riesgosos] <= 0).any():
    # con un p_i negativo los pesos proporcionales no tienen sentido
    print("OJO: hay un precio del riesgo negativo, así que se usan pesos iguales (1/3 cada uno)")
    w = pd.Series(1 / 3, index=riesgosos)
else:
    suma_p = p_riesgo[riesgosos].sum()
    w = p_riesgo[riesgosos] / suma_p
    print("w_i = p_i / (p_A + p_B + p_C)")
    for i in riesgosos:
        print(f"w_{i} = {p_riesgo[i]:.4f} / {suma_p:.4f} = {w[i]:.4f}")
print(f"w_A + w_B + w_C = {w.sum():.4f}")

r_P_s = w["A"] * rend["A"] + w["B"] * rend["B"] + w["C"] * rend["C"]
r_P = (pi_s * r_P_s).sum()
desv_P = r_P_s - r_P
var_P = (pi_s * desv_P ** 2).sum()
std_P = np.sqrt(var_P)

print(f"r_P,s = w_A r_A,s + w_B r_B,s + w_C r_C,s   (ejemplo {meses[0]}: "
      f"{w['A']:.4f} × {rend['A'].iloc[0]:.6f} + {w['B']:.4f} × {rend['B'].iloc[0]:.6f} + "
      f"{w['C']:.4f} × {rend['C'].iloc[0]:.6f} = {r_P_s.iloc[0]:.6f})")
print(f"r_P = (1/{N}) × {terminos(r_P_s)} = (1/{N}) × {r_P_s.sum():.6f} = {r_P:.6f} ({r_P:.4%})")
print(f"σ²_P = (1/{N}) × {terminos(desv_P ** 2)} = (1/{N}) × {(desv_P ** 2).sum():.6f} = {var_P:.6f}")
print(f"σ_P = √{var_P:.6f} = {std_P:.6f} ({std_P:.4%})")

# Si no hubiera diversificación, el riesgo de P sería el promedio ponderado de las σ
std_sin_div = (w * std[riesgosos]).sum()
menor = std[riesgosos].idxmin()
print(f"σ ponderada sin diversificar = Σ w_i σ_i = {w['A']:.4f} × {std['A']:.6f} + "
      f"{w['B']:.4f} × {std['B']:.6f} + {w['C']:.4f} × {std['C']:.6f} = {std_sin_div:.6f}")
print(f"Reducción de riesgo por diversificar: {std_sin_div:.4%} - {std_P:.4%} = {std_sin_div - std_P:.4%}")
print(f"Activo individual con menor riesgo: {menor} (σ = {std[menor]:.4%})  vs  σ_P = {std_P:.4%}\n")

# 8. Covarianza de A, B y C con el mercado
cov_m = {}
print("--- 8. Covarianza con el mercado ---")
print("Cov(R_i,R_m) = Σ π_s (r_i,s - r_i)(m_s - r_m)")
for i in riesgosos:
    prod = desv[i] * desv["M"]
    cov_m[i] = (pi_s * prod).sum()
    print(f"Cov(R_{i},R_m) = (1/{N}) × {terminos(prod)} = (1/{N}) × {prod.sum():.6f} = {cov_m[i]:.6f}")
cov_m = pd.Series(cov_m)
print()

# 9 a 13. Beta, cantidad de riesgo de mercado y CAPM
beta = cov_m / var["M"]
riesgo_mkt = beta * std["M"]
r_ajustado = r[riesgosos] - beta * (r["M"] - rf)
r_capm = rf + beta * (r["M"] - rf)
dif = r[riesgosos] - r_capm  # es lo mismo que r_ajustado - r_f

print("--- 9. Beta ---")
print("β_i = Cov(R_i,R_m) / Var(R_m)")
for i in riesgosos:
    print(f"β_{i} = {cov_m[i]:.6f} / {var['M']:.6f} = {beta[i]:.4f}")
print()

print("--- 10. Cantidad de riesgo de mercado ---")
print("β_i σ_m")
for i in riesgosos:
    print(f"β_{i} σ_m = {beta[i]:.4f} × {std['M']:.6f} = {riesgo_mkt[i]:.6f} ({riesgo_mkt[i]:.4%})")
print()

print("--- 11. Rendimiento ajustado por el riesgo vs r_f ---")
print(f"r_i - β_i (r_m - r_f)   ;   r_f = {rf:.6f}")
for i in riesgosos:
    comparacion = "mayor que r_f" if r_ajustado[i] > rf else "menor que r_f"
    print(f"{i}: {r[i]:.6f} - {beta[i]:.4f} × ({r['M']:.6f} - {rf:.6f}) = {r_ajustado[i]:.6f}  -> {comparacion}")
print()

print("--- 12. Rendimiento de equilibrio (CAPM) ---")
print("r_i* = r_f + β_i (r_m - r_f)")
for i in riesgosos:
    print(f"r_{i}* = {rf:.6f} + {beta[i]:.4f} × ({r['M']:.6f} - {rf:.6f}) = {r_capm[i]:.6f} ({r_capm[i]:.4%})")
print()

# Se redondea a 6 decimales para decidir si está "sobre" la recta,
# porque con todos los decimales la diferencia casi nunca da exactamente 0
posicion = {}
print("--- 13. Observado vs CAPM ---")
print("r_i - r_i*")
for i in riesgosos:
    if round(dif[i], 6) > 0:
        posicion[i] = "Por encima de la recta de mercado"
    elif round(dif[i], 6) < 0:
        posicion[i] = "Por debajo de la recta de mercado"
    else:
        posicion[i] = "Sobre la recta de mercado"
    print(f"{i} ({tickers[i]}): {r[i]:.6f} - {r_capm[i]:.6f} = {dif[i]:+.6f} ({dif[i]:+.4%})  -> {posicion[i]}")
print()

# Tabla de resultados (para el documento)
resultados = pd.DataFrame({
    "r_i": r,
    "varianza": var,
    "sigma_i": std,
    "p_i": p_riesgo,
    "x_i": x_inv,
    "r_x": r_x,
    "beta_i": beta,
    "beta_sigma_m": riesgo_mkt,
    "r_ajustado": r_ajustado,
    "r_CAPM": r_capm,
    "r_i - r_CAPM": dif,
    "posicion": pd.Series(posicion),
})
resultados.loc["P", ["r_i", "varianza", "sigma_i"]] = [r_P, var_P, std_P]
resultados = resultados.loc[activos + ["P"]].round(6)
resultados.to_csv("tabla_resultados.csv", encoding="utf-8-sig")

print("--- Tabla de resultados (mensual, en decimales) ---")
print(resultados.drop(columns="posicion").to_string())

# Tabla con todas las operaciones por estado (va en anexos)
ops = pd.DataFrame({"pi_s": pi_s}, index=rend.index)
for i in activos:
    ops["r_" + i] = rend[i]
    ops["r_" + i + " - media"] = desv[i]
    ops["(r_" + i + " - media)^2"] = desv[i] ** 2
for i, j in pares + [("A", "M"), ("B", "M"), ("C", "M")]:
    ops["prod_" + i + j] = desv[i] * desv[j]
ops["r_P"] = r_P_s
ops["(r_P - media)^2"] = desv_P ** 2
ops.index = meses
ops.loc["Suma"] = ops.sum()
ops.to_csv("tabla_operaciones_por_estado.csv", encoding="utf-8-sig")

fuente = f"Fuente: elaboración propia con datos mensuales de Yahoo Finance ({meses[0]} a {meses[-1]})"

# Gráfica 1: recta presupuestaria de cada activo combinado con F
sig = np.linspace(0, std.max() * 1.2, 50)

plt.figure(figsize=(8, 5))
for i in activos:
    linea, = plt.plot(sig * 100, (rf + p_riesgo[i] * sig) * 100, label=f"F-{i}  (pendiente p_{i} = {p_riesgo[i]:.3f})")
    plt.scatter(std[i] * 100, r[i] * 100, color=linea.get_color())
    plt.text(std[i] * 100 + 0.1, r[i] * 100, f"{i} ({tickers[i]})", fontsize=9)
plt.scatter(0, rf * 100, color="black")
plt.text(0.1, rf * 100, "F", fontsize=9)
plt.axvline(sigma_x * 100, color="gray", linestyle="--", label="Riesgo objetivo σ_x = σ_M")
plt.title("Rectas presupuestarias al combinar cada activo con F")
plt.xlabel("Desviación típica mensual (%)")
plt.ylabel("Rendimiento esperado mensual (%)")
plt.grid(True, alpha=0.4)
plt.legend(fontsize=8)
plt.figtext(0.01, 0.01, fuente, fontsize=8)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("grafica_rectas_presupuestarias.png", dpi=200)

# Gráfica 2: recta de mercado
x_betas = np.linspace(min(0, beta.min()), max(beta.max(), 1.2) * 1.15, 50)
y_recta = (rf + x_betas * (r["M"] - rf)) * 100

plt.figure(figsize=(8, 5))
plt.plot(x_betas, y_recta, label="Recta de mercado: r_f + β(r_m - r_f)")
plt.scatter(0, rf * 100, color="gray", label="F (activo libre de riesgo)")
plt.scatter(1.0, r["M"] * 100, color="black", label="Mercado (S&P 500)")
for i in riesgosos:
    plt.scatter(beta[i], r[i] * 100, label=f"{i} ({tickers[i]})")
    plt.vlines(beta[i], r_capm[i] * 100, r[i] * 100, colors="gray", linestyles=":")
    plt.text(beta[i] + 0.02, r[i] * 100, f"{i} ({tickers[i]})", fontsize=9)
plt.title("Recta de mercado (CAPM)")
plt.xlabel("Beta (β)")
plt.ylabel("Rendimiento esperado mensual (%)")
plt.grid(True, alpha=0.4)
plt.legend(fontsize=8)
plt.figtext(0.01, 0.01, fuente, fontsize=8)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("grafica_recta_de_mercado.png", dpi=200)
plt.show()
