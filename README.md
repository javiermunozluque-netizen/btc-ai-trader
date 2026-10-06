# BTC AI Trader V1.2

## Qué es
Dashboard experimental para BTC/USDT con datos públicos de Binance. No requiere claves de API y no ejecuta órdenes. La interfaz actualiza automáticamente cada 30 segundos mientras la página esté abierta.

Incluye:
- Precio BTC consultado en vivo mediante API pública (refresco de 30 s)
- BTC 1H/4H
- EMA55/EMA200
- RSI, ATR, volumen relativo
- estructura HH/HL/LH/LL
- Open Interest y funding
- score técnico + derivados
- LONG/SHORT/WAIT
- entrada, stop, TP y R:R
- gráfico
- backtest inicial de 5 años para umbrales 6/7/8

## Ejecutar en ordenador
```bash
pip install -r requirements.txt
streamlit run app.py
```

Después abre la URL local que indique Streamlit.

## Publicarlo para iPhone
La app debe alojarse en un servicio que ejecute Python/Streamlit (por ejemplo Streamlit Community Cloud, Render o similar). Una vez desplegada tendrás una URL que puedes abrir desde Safari.

## Actualización en tiempo real
La cotización se consulta en cada refresco y los indicadores se recalculan con velas de Binance. “Tiempo real” aquí significa actualización periódica, no feed tick-a-tick; la frecuencia efectiva depende del alojamiento, conexión y disponibilidad de Binance. Los datos de derivados pueden tener una frecuencia distinta.

## Importante
El backtest es una primera aproximación: no incluye slippage real, ejecución intrabar completa, funding histórico integrado en cada operación, liquidaciones históricas ni optimización walk-forward. Sirve para comparar rápidamente los umbrales, no para validar rentabilidad.
