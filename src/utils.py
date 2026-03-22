"""
Módulo de utilidades para el análisis estadístico de la asignación 
de crédito público y contingente de la CFN.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filepath):
    """
    Carga el dataset CSV y estandariza los nombres de las columnas 
    para su uso en fórmulas de statsmodels.
    
    Parámetros:
    filepath (str): Ruta al archivo CSV.
    
    Retorna:
    pd.DataFrame: DataFrame con los nombres de columnas limpios.
    """
    df = pd.read_csv(filepath)
    # Renombrar columnas específicas
    df.rename(columns={
        'Log-Amount Granted': 'Log_Amount', 
        'Credit Type': 'Credit_Type', 
        'Operation Type': 'Operation_Type', 
        'Operation Status': 'Operation_Status'
    }, inplace=True)
    # Reemplazar espacios por guiones bajos en el resto de columnas
    df.columns = df.columns.str.replace(' ', '_')
    return df

def fit_anova_5_factors(df):
    """
    Ajusta el modelo ANOVA multifactorial de 5 efectos principales.
    
    Parámetros:
    df (pd.DataFrame): Dataset limpio.
    
    Retorna:
    statsmodels.regression.linear_model.RegressionResultsWrapper: Modelo ajustado.
    """
    formula = 'Log_Amount ~ C(Region) + C(Sector) + C(Credit_Type) + C(Operation_Type) + C(Operation_Status)'
    model = ols(formula, data=df).fit()
    return model

def check_anova_assumptions(model, df):
    """
    Ejecuta y muestra en consola las pruebas estadísticas formales 
    para los supuestos de Normalidad y Homocedasticidad.
    
    Parámetros:
    model: Modelo ajustado de statsmodels.
    df (pd.DataFrame): Dataset limpio usado para el modelo.
    """
    residuals = model.resid
    
    # 1. Prueba de Normalidad (Shapiro-Wilk)
    stat_sw, p_sw = stats.shapiro(residuals)
    print("--- Prueba de Normalidad ---")
    print(f"Shapiro-Wilk Test: Estadístico = {stat_sw:.4f}, p-valor = {p_sw:.4e}\n")
    
    # 2. Prueba de Homocedasticidad (Levene)
    print("--- Prueba de Homocedasticidad (Levene) ---")
    factors = ['Region', 'Sector', 'Credit_Type', 'Operation_Type', 'Operation_Status']
    for factor in factors:
        groups = [df[df[factor] == cat]['Log_Amount'] for cat in df[factor].unique()]
        stat_lev, p_lev = stats.levene(*groups)
        print(f"Levene Test para '{factor}': Estadístico = {stat_lev:.4f}, p-valor = {p_lev:.4f}")

def plot_diagnostics(model, save_path=None):
    """
    Genera gráficos de diagnóstico para evaluar los residuos del modelo.
    
    Parámetros:
    model: Modelo ajustado de statsmodels.
    save_path (str, opcional): Ruta para guardar la imagen (ej. 'img/plot.png').
    """
    residuals = model.resid
    fitted = model.fittedvalues
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Residuos vs Valores Ajustados (Homocedasticidad)
    plt.subplot(1, 2, 1)
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuos vs Valores Ajustados\n(Modelo 5 Factores)')
    plt.xlabel('Valores Ajustados')
    plt.ylabel('Residuos')
    
    # Gráfico Q-Q (Normalidad)
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Gráfico Q-Q\n(Modelo 5 Factores)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"\nGráficos de diagnóstico guardados exitosamente en '{save_path}'")
        
    plt.show()