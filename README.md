# Pre-entrega 1: Pipeline de Entrenamiento y Clasificador Base

Pipeline base en PyTorch desarrollado para el Módulo 1 de Data Science III (Coderhouse). Implementa un ciclo de entrenamiento completo (forward, backward pass, optimización con Adam) y evaluación sobre partición de validación.

---

## 🛠️ Entorno y Dependencias
* **Python:** 3.10+
* **PyTorch:** 2.11.0 (CPU)
* **Scikit-Learn:** 1.6.1
* **NumPy:** 2.1.3
* **Dispositivo:** Detección dinámica y portable (`cuda` -> `mps` -> `cpu`)
* **Semilla de reproducibilidad:** `torch.manual_seed(42)`

---

## 🚀 Instalación y Ejecución

### Opción A: Google Colab
Puedes abrir y ejecutar el notebook directamente en Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gallimatias7/pre-entrega-1-nlp-gallimatias/blob/main/notebooks/pipeline.ipynb)

### Opción B: Ejecución Local
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/gallimatias7/pre-entrega-1-nlp-gallimatias.git](https://github.com/gallimatias7/pre-entrega-1-nlp-gallimatias.git)
   cd pre-entrega-1-nlp-gallimatias
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   ```

3. Instalar las dependencias exactas:
   ```bash
   pip install -r requirements.txt
   ```

4. Abrir y ejecutar el notebook:
   ```bash
   jupyter notebook notebooks/pipeline.ipynb
   ```

---

## 📊 Hiperparámetros y Experimento
* **Dataset:** 1000 muestras sintéticas (10 variables, 2 clases) escaladas con `StandardScaler`.
* **Arquitectura:** Red secuencial de 2 capas densas con activación intermedia ReLU y salida Sigmoid.
* **Función de Pérdida:** `BCELoss`
* **Optimizador:** Adam (`learning_rate = 0.01`)
* **Épocas:** 50

### Interpretación de la Curva de Pérdida
* **Train Loss:** Descendió de `0.5049` (época 10) a `0.2869` (época 50), demostrando una convergencia efectiva del modelo.
* **Val Loss / Val Acc:** La pérdida en validación bajó hasta `0.3840` en la época 30, logrando una exactitud máxima del 84.00%. A partir de la época 30, la pérdida de validación subió ligeramente hasta `0.3962` hacia la época 50, lo que evidencia un sobreajuste leve e indica que la época 30 representaba el punto óptimo para aplicar parada temprana (*early stopping*).
