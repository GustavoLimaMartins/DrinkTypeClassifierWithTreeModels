# ☕ Machine Learning - Classificação de Bebidas pelo Teor de Cafeína

Este projeto utiliza **Machine Learning** para classificar diferentes tipos de bebidas (café, chá, refrigerante, energéticos, etc.) com base em seus atributos nutricionais e de volume.  
O dataset utilizado (`caffeine.csv`) contém informações como volume, calorias e cafeína, sendo usado para treinar e avaliar modelos de classificação.

---

## 📂 Estrutura do Projeto

- `caffeine.csv` → Dataset com informações das bebidas.  
- `decision_tree.py` → Script Python que realiza a análise exploratória dos dados, treinamento e avaliação dos modelos.  

---

## 📊 Dataset

O dataset inclui os seguintes atributos principais:

- **Volume (ml)** → Quantidade da bebida.  
- **Calories** → Valor calórico.  
- **Caffeine (mg)** → Quantidade de cafeína.  
- **type** → Categoria da bebida (*target*).  
- **drink** → Nome específico da bebida.  

---

## 🔎 Análise Exploratória

O script realiza:
- Estatísticas descritivas dos dados.  
- Histogramas dos atributos numéricos.  
- `pairplot` para visualizar relações entre atributos.  
- Heatmap da correlação entre variáveis.  

---

## 🤖 Modelos de Machine Learning

Foram aplicados três algoritmos principais para classificação:

1. **Árvore de Decisão (Decision Tree Classifier)**  
   - Profundidade máxima: 4  
   - Critério: `gini`  
   - Exibição gráfica da árvore de decisão  

2. **Random Forest Classifier**  
   - Oversampling aplicado com **SMOTE** para balanceamento das classes minoritárias  
   - Máx. profundidade: 7  
   - Nº de estimadores: 80  
   - Critério: `entropy`  
   - Classes balanceadas  

3. **Support Vector Machine (SVM)**  
   - Kernel padrão  
   - Treinado após oversampling  

---

## 📈 Resultados Obtidos

- **Árvore de Decisão (DT):** ~65% de acurácia  
- **Random Forest (RF):** ~72% de acurácia  
- **SVM:** ~63% de acurácia  

---
