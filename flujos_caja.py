import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

# Flujos de caja neto por escenario
flujos_conservador = [
    -387785,
    -217599,
    12108,
    270868,
    609254,
    1006920
]

flujos_probable = [
    -387785,
    -164412,
    133677,
    468417 ,
    903045 ,
    1412149 
    ]

flujos_optimo = [
    -387785,
    -111226,
    255246,
    665967,
    1196837,
    1817380
]

anios = [0, 1, 2, 3, 4, 5]

import matplotlib.pyplot as plt

def plot_flujos(flujos, titulo):
    anios = [f"Año {i}" for i in range(len(flujos))]
    colores = ['red' if f < 0 else 'blue' for f in flujos]

    fig, ax = plt.subplots(figsize=(8, 5))
    

    bars = ax.bar(anios, flujos, color=colores)

    # Etiquetas numéricas dentro o justo sobre la barra
    for bar, valor in zip(bars, flujos):
        altura = bar.get_height()
        offset = 0.02 * max(abs(f) for f in flujos)

        if valor >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                altura - offset if altura > offset else altura + offset,
                f"{valor:,.0f} €",
                ha='center',
                va='bottom' if altura <= offset else 'top',
                fontsize=9,
                color='black' if altura <= offset else 'black'
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                altura + offset,
                f"{valor:,.0f} €",
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )

    ax.set_title(titulo)
    ax.set_xlabel('Año')
    ax.set_ylabel('Flujo de Caja (€)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



# Mostrar gráficos uno por uno
plot_flujos(flujos_conservador, 'Escenario Conservador (20%)')
plot_flujos(flujos_probable, 'Escenario Probable (25%)')
plot_flujos(flujos_optimo, 'Escenario Óptimo (30%)')

# Cálculo e impresión de VAN y TIR
tasa_descuento = 0.24  # 8%
escenarios = {
    "Conservador": flujos_conservador,
    "Probable": flujos_probable,
    "Óptimo": flujos_optimo
}

print("\n--- Resultados Financieros ---")
for nombre, flujos in escenarios.items():
    van = npf.npv(tasa_descuento, flujos)
    tir = npf.irr(flujos)
    print(f"{nombre}: VAN = {van:,.2f} €, TIR = {tir*100:.2f} %")

