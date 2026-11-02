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
