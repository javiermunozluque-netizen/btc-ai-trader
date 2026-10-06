# BTC AI Trader V1.5

## Qué es
Dashboard experimental para BTC/USDT con datos públicos de Binance. No requiere claves de API, no conecta cuentas y no ejecuta órdenes.

## Funciones
- Cotización BTC/USDT y velas 1H/4H cerradas
- EMA55/EMA200, RSI, ATR, volumen relativo y estructura HH/HL/LH/LL
- Open Interest y funding como contexto de derivados
- Score heurístico LONG/SHORT/WAIT y niveles ilustrativos
- Gráfico de precio e indicadores
- Validación cronológica V1.4: comparación de umbrales en desarrollo y evaluación del seleccionado en el tramo OOS
- Laboratorio V1.5: compara filtro base, tendencia EMA, momentum, volumen y estructura usando el mismo umbral fijo
- Auditoría de operaciones OOS: dirección LONG/SHORT, precio de entrada y salida, motivo de salida, R bruto, costes estimados, R neto y duración
- Resumen de resultados por dirección y motivo de salida; exportación CSV de las operaciones

## Cómo interpretar V1.5
La comparación de filtros se hace en el tramo de desarrollo (70% del histórico) y el filtro con mejor expectativa se evalúa en el 30% final fuera de muestra. Como se comparan varias alternativas, la elección aún puede sufrir sobreajuste; el resultado OOS debe tratarse como exploratorio, no como validación definitiva. El umbral de comparación del laboratorio es fijo en 7 para comparar filtros en condiciones homogéneas.

El modelo es una aproximación 1H y no replica exactamente la señal multi-timeframe en vivo. Usa entrada en la apertura siguiente, stop de 1 ATR, objetivo de 2 ATR y, si stop y objetivo se tocan en la misma vela, contabiliza primero el stop. Los costes son estimaciones configurables, no ejecuciones reales. No incorpora funding histórico por operación, liquidez ni impacto de mercado.

El histórico está limitado a 12.000 velas de 1H (aproximadamente 16 meses), aunque la descarga solicite un periodo mayor. La rentabilidad pasada no garantiza resultados futuros; no utilizar como sistema probado para operar con dinero real.

## Ejecutar en ordenador
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar para iPhone
Aloja la app en un servicio compatible con Streamlit (por ejemplo, Streamlit Community Cloud) y abre la URL desde Safari.

## Datos y seguridad
La cotización se actualiza periódicamente; no es un feed tick-a-tick. La app usa endpoints públicos de Binance y no necesita claves privadas ni permisos de trading. Es una herramienta experimental, no asesoramiento financiero.
