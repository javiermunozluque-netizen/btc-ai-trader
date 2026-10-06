# BTC AI Trader V1.7

## Qué es
Dashboard experimental para BTC/USDT con datos públicos de Binance. No requiere claves de API, no conecta cuentas y no ejecuta órdenes.

## Funciones
- Cotización BTC/USDT y velas 1H/4H cerradas
- EMA55/EMA200, RSI, ATR, volumen relativo y estructura HH/HL/LH/LL
- Open Interest y funding como contexto de derivados
- Score heurístico LONG/SHORT/WAIT y niveles ilustrativos
- **Gráfico interactivo de velas japonesas** con EMA55, EMA200 y volumen; zoom, desplazamiento y hover de datos
- Validación cronológica: comparación de umbrales en desarrollo y evaluación del seleccionado en el tramo OOS
- Laboratorio: compara filtro base, tendencia EMA, momentum, volumen y estructura con umbral fijo
- Auditoría OOS: dirección, precios, motivo de salida, R bruto, costes, R neto y duración
- **Curva de R acumulado OOS** para ver la evolución de la estrategia en vez de depender solo de una tabla
- Resúmenes por dirección y motivo de salida y exportación CSV

## Cambios de diseño V1.7
- Sustituye el gráfico de líneas básico por velas japonesas interactivas
- Separa visualmente precio y volumen, con EMA55 y EMA200 superpuestas
- Añade una curva de resultados acumulados de operaciones OOS y métricas de drawdown
- Mantiene el diseño adaptable a móvil y el acento naranja inspirado en Bitcoin

## Cómo interpretar los resultados
La comparación de filtros se hace en el tramo de desarrollo (70% del histórico) y el filtro con mejor expectativa se evalúa en el 30% final fuera de muestra. Como se comparan varias alternativas, la elección puede sufrir sobreajuste; el resultado OOS es exploratorio, no validación definitiva. El umbral del laboratorio es fijo en 7 para comparar filtros en condiciones homogéneas.

El modelo es una aproximación 1H y no replica exactamente la señal multi-timeframe en vivo. Usa entrada en la apertura siguiente, stop de 1 ATR y objetivo de 2 ATR. Si stop y objetivo se tocan en la misma vela, contabiliza primero el stop. Los costes son estimaciones configurables, no ejecuciones reales. No incorpora funding histórico por operación, liquidez ni impacto de mercado.

El histórico está limitado a las últimas 12.000 velas de 1H (aproximadamente 16 meses). La evaluación OOS conserva el historial necesario para calcular indicadores, pero solo permite abrir operaciones a partir del corte cronológico. La rentabilidad pasada no garantiza resultados futuros; no utilizar como sistema probado para operar con dinero real.

## Ejecutar en ordenador
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar para iPhone
Aloja la app en un servicio compatible con Streamlit (por ejemplo, Streamlit Community Cloud) y abre la URL desde Safari.

## Datos y seguridad
La app usa endpoints públicos de Binance y no necesita claves privadas ni permisos de trading. Es una herramienta experimental, no asesoramiento financiero.
