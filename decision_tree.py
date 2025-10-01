import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
# Conversão da base de dados para formato matricial
df = pd.read_csv('Modelos ML/Árvores de Decisão/caffeine.csv')
print('Dimensões:', df.shape)
print(f'Primeiras 5 linhas:\n{df.head()}')
print(f'Topologia dos dados:\n{df.info()}')
print(f'Estatística básica aplicada:\n{df.describe()}')
# Plotagem do histograma por atributo numérico
df.hist()
plt.show()
# Plotagem para análise do comportamento dos atributos em pares
sb.pairplot(df, vars=['Volume (ml)', 'Calories', 'Caffeine (mg)'], hue='type')
plt.show()
# Plotagem da correlação dos atributos nominais categóricos
fig, ax = plt.subplots(figsize=(8,8))
sb.heatmap(data=df.corr(numeric_only=True).round(2), annot=True, linewidths=.5, ax=ax)
plt.show()

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
X = df.drop(columns=['type', 'drink'])
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Modelo de classificação baseado em Árvore de Decisão
dt = DecisionTreeClassifier(max_depth=4, random_state=7, criterion='gini')
dt.fit(X_train_scaler, y_train)
dt_pred = dt.predict(X_test_scaler)

class_names = ['Coffee', 'Energy Drinks', 'Energy Shots', 'Soft Drinks', 'Tea', 'Water']
label_names = ['Volume (ml)', 'Calories', 'Caffeine (mg)']
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,15), dpi=300)
plot_tree(dt, feature_names = label_names, class_names=class_names, filled = True)
plt.show()

print(f'Acurácia DT: {accuracy_score(y_test, dt_pred)*100:.2f}%')

# Técnica de oversampling (maximização de instâncias das classes minoritárias)
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train_os, y_train_os = oversample.fit_resample(X_train_scaler, y_train)

# Modelo de classificação baseado em Random Forest
rf = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=7, n_estimators=80, random_state=7)
rf.fit(X_train_os, y_train_os)
rf_pred = rf.predict(X_test_scaler)

print(f'Acurácia RF: {accuracy_score(y_test, rf_pred)*100:.2f}%')

# Modelo de classificação baseado em Máquinas de Vetores de Suporte (SVM)
svm = SVC(random_state=7)
svm.fit(X_train_os, y_train_os)
svm_pred = svm.predict(X_test_scaler)
print(f'Acurácia SVM: {accuracy_score(y_test, svm_pred)*100:.2f}%')
