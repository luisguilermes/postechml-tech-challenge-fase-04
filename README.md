## Tech Challenge - Fase 04

Projeto: previsão do preço de fechamento de ações usando um modelo LSTM + API para inferência.

Este repositório contém um scaffold completo para: coleta de dados, pré-processamento,
treinamento rápido de um LSTM em PyTorch, e uma API em FastAPI para servir previsões.

Principais arquivos e pastas

- [requirements.txt](requirements.txt) — dependências Python.
- [src/](src) — código (data_fetch, models, api, train).
- [scripts/fetch_sample.py](scripts/fetch_sample.py) — baixar amostra via `yfinance`.
- [Dockerfile](Dockerfile) — imagem para rodar a API com os modelos salvos.
- [data/](data) — (não comitado) datasets baixados pelo script.
- [models/](models) — modelos e scalers gerados pelo treinamento.

Pré-requisitos

- Python 3.10+
- Docker (opcional, para deploy)

Instalação local

1. Criar ambiente virtual e instalar dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Baixar dados de exemplo (ex.: `BBAS3.SA`):

```bash
PYTHONPATH=. python3 scripts/fetch_sample.py --symbol BBAS3.SA --start 2018-01-01 --end 2024-07-20
```

Treinamento rápido

Um treinamento mínimo está disponível em `src/train.py`. Exemplo de execução (teste rápido):

```bash
PYTHONPATH=. python3 src/train.py --csv data/BBAS3.SA.csv --epochs 5 --window 20 --batch_size 64
```

Isso gera:

- `models/lstm_<symbol>.pt` — pesos do modelo
- `models/scaler_<symbol>.pkl` — scaler usado para normalização

Avaliação

Após treinar, utilize `src/evaluate.py` para avaliar o modelo na base de validação. O script calcula as métricas:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

Exemplo de execução:

```bash
PYTHONPATH=. python3 src/evaluate.py --csv data/BBAS3.SA.csv \
	--model models/lstm_BBAS3.SA.pt --scaler models/scaler_BBAS3.SA.pkl --window 20 --test_size 0.2
```

Saídas geradas (pasta `reports/`):

- `reports/metrics_<symbol>.json` — JSON com MAE, RMSE, MAPE.
- `reports/pred_vs_true_<symbol>.png` — gráfico previsões vs reais na base de validação.

Exemplo de métricas geradas (resultado do modelo de `BBAS3.SA` deste scaffold):

- MAE: ~3.2463
- RMSE: ~3.6833
- MAPE: ~13.90%

Use essas métricas para comparar variantes do modelo (diferentes janelas, hiperparâmetros, features).

API de inferência

Start local (desenvolvimento):

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoint principal:

- `POST /predict` — corpo JSON com `symbol` e `history` (lista de últimos preços de fechamento, tamanho >= `window`)

Exemplo curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
	-d '{"symbol":"BBAS3.SA","history":[24.48,24.56,24.81,24.70,24.63, ... 20 valores ]}'
```

Docker (rodar a API com modelos incluídos)

Para gerar a imagem (inclui `models/` do workspace):

```bash
docker build -t stock-lstm:local .
docker run -d --name stock-lstm-local -p 8000:8000 stock-lstm:local
```

Observações importantes

- O endpoint `POST /predict` espera pelo menos `window` (padrão 20) valores em `history`.
- O scaffold salva modelos e scaler em `models/` — quando treinar em outro símbolo, copie/nomine adequadamente.
- Arquitetura e hiperparâmetros são básicos; recomenda-se tuning para produção.

Próximos passos / melhorias sugeridas

- Salvar metadados de treino (window, hidden_size, num_layers) junto ao modelo.
- Endpoints para listar modelos disponíveis e exibir metadados.
- Monitoramento básico (logging, métricas Prometheus) para produção.
- Notebook Jupyter com análise exploratória e gráficos de previsão vs real.

Contato

Se quiser, eu posso: (1) adicionar endpoint de listagem de modelos, (2) salvar metadados no treino, ou (3) criar notebooks e relatórios de avaliação.
