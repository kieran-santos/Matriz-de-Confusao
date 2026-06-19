# Regressão Logística - Matriz de Confusão

Trabalho desenvolvido para a disciplina de **Machine Learning**, com o objetivo de implementar uma **regressão logística do zero**, sem o uso de bibliotecas de Machine Learning (como `scikit-learn`), apenas com `numpy`, avaliando o modelo através de uma **matriz de confusão** e suas métricas derivadas.

O projeto utiliza o famoso dataset do **Titanic** para prever, a partir de características dos passageiros, se eles sobreviveram ou não ao naufrágio.

## Estrutura do projeto

```
.
├── regre_logistica_mconfusao.ipynb   # Implementação da regressão logística com avaliação por matriz de confusão
└── titanic.csv                        # Dataset com dados dos passageiros do Titanic
```

## Notebook

### `regre_logistica_mconfusao.ipynb`

Implementação completa de um classificador binário usando **regressão logística** treinada por **gradiente descendente**, com foco em avaliar o modelo de forma mais completa que a simples acurácia:

#### 1. Leitura dos dados
- Leitura do `titanic.csv` com o módulo `csv` (`DictReader`)
- Seleção das features: `pclass`, `sex`, `age`, `sibsp`, `parch` e `fare`
- Conversão da coluna `sex` para valor numérico (`1` para `female`, `0` para `male`)
- Linhas sem o valor de `survived` (target) são descartadas

#### 2. Tratamento de dados
- Preenchimento de idades ausentes (`fill_missing_age`) com a **média das idades conhecidas**
- Normalização das features (`normalize`) subtraindo a média e dividindo pelo desvio padrão, para evitar que variáveis em escalas diferentes (ex: `fare` vs `sex`) distorçam o treinamento
- Adição da coluna de **bias/intercepto** (`add_bias`) à matriz X

#### 3. Funções do modelo
- **Função sigmoide**, que transforma a saída linear em uma probabilidade entre 0 e 1
- **Função de custo** (entropia cruzada / log loss), usada para medir o erro do modelo
- **Gradiente descendente**, que ajusta os pesos (`beta`) a cada época para minimizar o custo

#### 4. Treinamento
- Pesos iniciados em zero
- Treinamento por **2000 épocas** com taxa de aprendizado (`lr`) de `0.01`

#### 5. Predição e Matriz de Confusão
- Previsão da classe (sobreviveu / não sobreviveu) usando um **decision boundary** de `0.5`
- Cálculo manual da **matriz de confusão**, contabilizando:
  - **VP** (Verdadeiros Positivos)
  - **VN** (Verdadeiros Negativos)
  - **FP** (Falsos Positivos)
  - **FN** (Falsos Negativos)
- Cálculo das métricas derivadas da matriz de confusão:
  - **Acurácia** — proporção geral de acertos
  - **Precisão** — dos previstos como sobreviventes, quantos realmente sobreviveram
  - **Revocação (Recall)** — dos que realmente sobreviveram, quantos o modelo identificou
  - **Especificidade** — dos que realmente não sobreviveram, quantos o modelo identificou corretamente
  - **F1 Score** — média harmônica entre precisão e revocação

#### 6. Split treino/teste
- Divisão manual do dataset em treino (80%) e teste (20%), com embaralhamento (`np.random.shuffle`) e seed fixa (`42`) para reprodutibilidade
- O conjunto de teste passa pelas **mesmas transformações** (preenchimento de idade, normalização e bias) aplicadas ao treino

### Resultado obtido

Com os parâmetros usados (2000 épocas, `lr=0.01`), o modelo obteve no conjunto de teste:

| Métrica | Valor |
|---|---|
| Acurácia | 72,9% |
| Precisão | 68,0% |
| Revocação | 63,6% |
| Especificidade | 79,4% |
| F1 Score | 65,7% |

## Tecnologias utilizadas

- Python 3
- [NumPy](https://numpy.org/)
- `csv` (módulo padrão do Python)

## Como executar

> Este notebook foi originalmente desenvolvido no **Google Colab**, lendo o `titanic.csv` a partir do Google Drive (`drive.mount`). Para rodar localmente (Jupyter), remova/ajuste a célula de `drive.mount` e aponte `load_data` diretamente para o arquivo `titanic.csv` na pasta do projeto.

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/regressao-logistica-matriz-confusao.git
   cd regressao-logistica-matriz-confusao
   ```

2. Instale as dependências:
   ```bash
   pip install numpy
   ```

3. Abra o notebook com Jupyter:
   ```bash
   jupyter notebook
   ```

4. Execute as células de `regre_logistica_mconfusao.ipynb` em ordem (ajustando o caminho do `titanic.csv` se estiver rodando localmente, conforme observação acima).

> O arquivo `titanic.csv` precisa estar acessível no caminho informado em `load_data()` para que a leitura dos dados funcione corretamente.

## Sobre o dataset `titanic.csv`

Dataset clássico de Machine Learning contendo informações sobre os passageiros do Titanic, como classe da passagem (`pclass`), sexo, idade, número de irmãos/cônjuges a bordo (`sibsp`), número de pais/filhos a bordo (`parch`), tarifa paga (`fare`) e se o passageiro sobreviveu (`survived`). Neste projeto, a coluna `survived` é usada como variável alvo (y) e as demais como variáveis preditoras (X).

## Observações

Este projeto tem fins **didáticos**, com foco em compreender o funcionamento matemático da regressão logística (função sigmoide, função de custo de entropia cruzada e otimização via gradiente descendente), além de aprofundar a **avaliação de modelos de classificação** além da acurácia simples, através da matriz de confusão e suas métricas derivadas (precisão, revocação, especificidade e F1 Score). Todos os cálculos foram implementados manualmente, sem o uso de bibliotecas prontas de Machine Learning.
