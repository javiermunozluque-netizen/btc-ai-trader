# BTC AI Trader V1.9

## Qué es
Dashboard experimental para BTC/USDT con datos públicos de Binance. No requiere claves de API, no conecta cuentas y no ejecuta órdenes.

## Funciones
- Cotización BTC/USDT y velas 1H/4H cerradas
- EMA55/EMA200, RSI, ATR, volumen relativo y estructura HH/HL/LH/LL
- Open Interest y funding como contexto de derivados
- Score heurístico LONG/SHORT/WAIT y niveles ilustrativos
- Gráfico interactivo de velas japonesas con EMA55, EMA200 y volumen
- Validación cronológica con comparación de umbrales y tramo OOS
- Laboratorio que compara cinco filtros técnicos y tres modos de dirección: ambos, LONG solamente y SHORT solamente
- Gráfico horizontal de expectativa neta por configuración para detectar rápidamente configuraciones débiles
- Selección del candidato solo en desarrollo, priorizando al menos 15 operaciones (fallback a 5 si no hay candidatos); evaluación posterior en el 30% final OOS
- Curva de R acumulado, drawdown, resumen por dirección y motivo de salida
- Auditoría y exportación CSV de las operaciones OOS

## Cambios de V1.9
- Evalúa LONG y SHORT por separado para comprobar si un sentido contribuye desproporcionadamente a las pérdidas
- Compara las combinaciones de filtros técnicos con ambos sentidos y modos de dirección restringidos
- Añade un gráfico de barras de expectativa neta por configuración y mantiene la curva acumulada OOS
- Evita priorizar candidatos con menos de 15 operaciones de desarrollo cuando existe una muestra mayor suficiente; si no, relaja el umbral a 5 y deja constancia de que la evidencia es limitada
- Mantiene el gráfico de velas interactivo y el diseño responsive de V1.7

## Cómo interpretar los resultados
La selección de una configuración se realiza exclusivamente en el tramo de desarrollo (70% inicial). Se evalúa después en el 30% final OOS. Probar múltiples filtros puede producir sobreajuste incluso con esta separación; el resultado es exploratorio y no valida rentabilidad futura. No operar en real basándose solo en este backtest.

El modelo es una aproximación 1H y no replica exactamente la señal multi-timeframe en vivo. Usa entrada en la apertura siguiente, stop de 1 ATR y objetivo de 2 ATR. Si stop y objetivo se tocan en la misma vela, contabiliza primero el stop. Los costes son estimaciones configurables. No incorpora funding histórico por operación, liquidez ni impacto de mercado.

El histórico está limitado a las últimas 12.000 velas de 1H (aproximadamente 16 meses). La evaluación OOS conserva el historial para calcular indicadores, pero solo permite abrir operaciones a partir del corte cronológico.

## Ejecutar en ordenador
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar para iPhone
Aloja la app en un servicio compatible con Streamlit (por ejemplo, Streamlit Community Cloud) y abre la URL desde Safari.

## Datos y seguridad
La app usa endpoints públicos de Binance y no necesita claves privadas ni permisos de trading. Es una herramienta experimental, no asesoramiento financiero.


### V1.9 — auditoría de backtest
- Auditoría explícita de salidas de 0h y operaciones que terminan en la misma vela.
- Detección de velas ambiguas cuando OHLC toca simultáneamente stop y objetivo; se aplica stop-first.
- Desglose por operación de comisión, deslizamiento y coste total en USDT y R.
- Sensibilidad del resultado OOS a costes de 0–20 pb ida+vuelta.
- Mantiene separación desarrollo/OOS y selección de dirección/filtro solo en desarrollo.
