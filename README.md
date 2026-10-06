# BTC AI Trader V1.4

## Qué es
Dashboard experimental para BTC/USDT con datos públicos de Binance. No requiere claves de API, no conecta cuentas y no ejecuta órdenes. La interfaz se actualiza periódicamente mientras la página está abierta.

## Funciones
- Cotización pública BTC/USDT y velas 1H/4H cerradas
- EMA55/EMA200, RSI, ATR, volumen relativo y estructura HH/HL/LH/LL
- Open Interest y funding como contexto de derivados
- Score heurístico LONG/SHORT/WAIT y niveles ilustrativos de entrada, stop y objetivo
- Gráfico de precio e indicadores
- Backtest 1H con comisión y deslizamiento configurables
- Validación cronológica V1.4: compara umbrales en el tramo inicial de desarrollo (70%) y evalúa el umbral seleccionado en el último 30% reservado fuera de muestra
- Exportación CSV de las métricas de desarrollo y de la prueba OOS

## Cómo interpretar la validación
El umbral se elige por expectativa R en el tramo de desarrollo; la muestra OOS no se utiliza para elegirlo. Los indicadores se calculan sobre el histórico anterior disponible y el backtest OOS comienza en el punto de corte tras un periodo de calentamiento de 300 velas. Si la expectativa OOS es negativa o nula, el sistema no supera esa prueba.

El histórico descargado está limitado a 12.000 velas de 1H (aproximadamente 16 meses), no a cinco años completos. La validación es un primer filtro, no una prueba definitiva: el modelo 1H no replica exactamente la señal multi-timeframe en vivo, y no incluye funding histórico por operación, liquidez, impacto de mercado ni calidad real de ejecución. La selección entre tres umbrales puede seguir sobreajustándose al tramo de desarrollo; conviene probar otros periodos y mercados antes de cualquier uso real.

## Ejecutar en ordenador
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar para iPhone
Aloja la app en un servicio compatible con Streamlit (por ejemplo, Streamlit Community Cloud) y abre la URL desde Safari.

## Datos y seguridad
La cotización se actualiza periódicamente; no es un feed tick-a-tick. La disponibilidad de datos depende de Binance, la conexión y el alojamiento. Esta herramienta es experimental, no constituye asesoramiento financiero y no debe considerarse un sistema probado para operar con dinero real.
