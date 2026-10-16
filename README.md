# fin_apps
Financial apps for personal use

## Preço justo de opções (`app.py`, `options_app/`)

App local para **vender opções** sobre uma carteira de longo prazo (B3 e EUA). Não é plataforma de arbitragem: o preço
justo serve de referência para o preço da ordem e para comparar strikes e vencimentos por rendimento × probabilidade
de exercício.

- **Carteira**: o Claude lê os extratos da BTG, XP, Avenue e Nomad (PDF, print, CSV ou Excel) e grava as posições em
  `data/carteira.json`. A tabela pode ser editada no app. Cada novo extrato substitui as posições daquela corretora.
- **Calls cobertas**: preço justo das calls sobre cada ativo com pelo menos 1 lote livre (100 ações). Ações que já
  cobrem calls lançadas ficam de fora. A sugestão é o strike/vencimento de maior rendimento anualizado dentro da
  probabilidade máxima de exercício, com strike ≥ preço médio (opcional).
- **Puts para comprar**: digite `PLTR`, `PETR4:34` ou `VALE3:55:300` (ticker[:preço alvo[:quantidade]]). Mostra o
  preço justo da put, o preço efetivo de compra se exercido (strike − prêmio), o caixa necessário e se o alvo é atingido.
- **Calculadora**: uma opção específica, com gregas e volatilidade implícita.

### Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # para ler extratos com o Claude (ou cole a chave na barra lateral)
streamlit run app.py            # http://localhost:8501
```

Para testar sem internet, escolha **Demo** na barra lateral e use **Carregar carteira de exemplo** (só aparece com a
carteira vazia).

### Como o preço justo é calculado

| Dado | Fonte |
|---|---|
| Cotação, histórico, dividendos | Yahoo Finance (`yfinance`), `PETR4` → `PETR4.SA` |
| Volatilidade | Histórica: mista (padrão, média de 60 pregões e 1 ano), janela simples ou EWMA |
| Juro livre de risco | B3: Selic meta (API SGS do Banco Central, série 432). EUA: T-bill de 13 semanas (`^IRX`). Ambos podem ser definidos à mão |
| Modelo | Árvore binomial CRR (300 passos, exercício americano) com dividend yield. Black-Scholes para gregas e probabilidade |
| Strikes / vencimentos | EUA: série listada no Yahoo, com compra/venda de mercado como referência. B3: o Yahoo não tem cadeia de opções da B3, então os strikes são teóricos (grade de valores redondos) nos vencimentos mensais (3ª sexta-feira). Confira a série existente no home broker |

O lote padrão é de 100 ações nos dois mercados. Não é recomendação de investimento.

### Linha de comando

```bash
python -m options_app.cli ler-extratos extratos/btg.pdf --corretora "BTG Pactual" --salvar
python -m options_app.cli calls            # sugestões (use --todos para ver todos os strikes)
python -m options_app.cli puts "PLTR:40, PETR4:34"
python -m options_app.cli preco --spot 37.5 --strike 40 --dias 30 --vol 0.3 --tipo call --juro 0.13
```

### Testes

```bash
python -m pytest
```
