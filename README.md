# Análisis Estadístico de Asignación de Crédito Público - CFN Ecuador (2025)

Este repositorio contiene el análisis de datos, código fuente y justificación metodológica para el estudio de la asignación de crédito público y contingente por parte de la Corporación Financiera Nacional (CFN) del Ecuador, correspondiente al período enero-septiembre de 2025. 

El objetivo principal de este proyecto es modelar los determinantes del volumen de crédito otorgado para sustentar la redacción de un artículo científico sobre política crediticia pública.

## 📊 Descripción del Proyecto

El análisis se centra en modelar la variable respuesta continua (monto del crédito otorgado) en función de diversas variables macroestructurales y operativas. Dada la asimetría positiva típica de los datos financieros, se aplicó una transformación logarítmica, trabajando con un modelo log-normal para la variable `Log-Amount Granted`.

### Datos
El dataset principal (`cleaned_cfn_volumencreditocontingente_2025_enero-septiembre.csv`) incluye operaciones crediticias categorizadas por:
* **Region:** Amazonía, Costa, Insular, Sierra.
* **Sector:** Primario, Secundario, Terciario, Cuaternario.
* **Credit Type:** Pymes, Corporativo, Microcrédito, Empresarial.
* **Operation Type:** Crédito, Contingente, Factoring.
* **Operation Status:** Original, Reestructurado, Novado, Refinanciado.

## 🔬 Metodología Estadística

El enfoque metodológico transitó por una evaluación rigurosa de supuestos para garantizar estimadores insesgados:

1. **Evaluación de ANCOVA (Descartado):** Inicialmente se evaluó un Análisis de la Covarianza (ANCOVA) utilizando el "Número de Operaciones" como covariable continua. Este modelo fue descartado debido a la violación del supuesto crítico de homogeneidad de pendientes (interacción significativa entre la Región y el Número de Operaciones, $p = 0.0047$), además de presentar una correlación lineal general muy débil ($r = 0.1082$).
2. **Modelo Seleccionado (ANOVA Multifactorial de 5 Factores):** Se optó por un Análisis de Varianza estándar eliminando la covariable empírica. El modelo final evalúa los efectos principales de las 5 variables categóricas estructurales de la base de datos.

### Validación de Supuestos del Modelo Final
El modelo de 5 factores demostró una alta robustez estadística, cumpliendo satisfactoriamente con los supuestos paramétricos:
* **Normalidad de Residuos:** Verificado mediante la prueba de Shapiro-Wilk ($p = 0.1993$).
* **Homocedasticidad:** Confirmada mediante la prueba de Levene para todos los factores ($p > 0.05$ en todos los cortes categóricos).
* **Independencia:** Asumida por el diseño transversal de la base de datos de la institución.

## 🛠️ Tecnologías y Requisitos

El análisis fue desarrollado en Python. Las principales librerías utilizadas son:
* `pandas` (Manipulación de datos)
* `numpy` (Computación numérica)
* `statsmodels` (Modelado estadístico y pruebas ANOVA)
* `scipy` (Pruebas de hipótesis y supuestos)
* `matplotlib` & `seaborn` (Visualización de datos y gráficos de diagnóstico)

## 📂 Estructura del Repositorio

```text
├── Data/
│   └── cleaned_cfn_volumencreditocontingente_2025_enero-septiembre.csv
├── Notebooks/
│   └── ANCOVA.ipynb
│   └── ANCOVA_II.ipynb
│   └── ...
├── Figures/
│   ├── anova_5_diagnostic_plots.png
│   └── ...
├── src/
│   └── utils.py
├── README.md
└── requirements.txt
