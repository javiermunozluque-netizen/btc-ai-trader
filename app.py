
import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="BTC AI Trader V1.5", page_icon="₿", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(70,85,110,.12),rgba(70,85,110,.04));border:1px solid rgba(128,128,128,.22);padding:16px 18px;border-radius:14px}
[data-testid="stMetricLabel"]{font-size:.85rem}
[data-testid="stMetricValue"]{font-weight:700}
section[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.2)}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:10px 16px}
div[data-testid="stAlert"]{border-radius:12px}
hr{margin:1.5rem 0}
</style>""",unsafe_allow_html=True)

SPOT="https://data-api.binance.vision"
FUT="https://fapi.binance.com"
TIMEOUT=15

@st.cache_data(ttl=20)
def get_json(url, params=None):
    r=requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def klines(symbol="BTCUSDT", interval="1h", limit=1000, start=None, end=None):
    p={"symbol":symbol,"interval":interval,"limit":min(limit,1000)}
    if start is not None: p["startTime"]=int(start)
    if end is not None: p["endTime"]=int(end)
    data=get_json(SPOT+"/api/v3/klines",p)
    cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    df=pd.DataFrame(data,columns=cols)
    for c in cols[1:6]: df[c]=pd.to_numeric(df[c])
    df["open_time"]=pd.to_datetime(df.open_time,unit="ms",utc=True)
    return df[["open_time","open","high","low","close","volume","quote_volume","trades"]]

def paginate_klines(symbol, interval, start_ms, end_ms, max_rows=12000):
    out=[]; cur=start_ms
    step={"1h":3600000,"4h":14400000}[interval]
    while cur < end_ms and sum(len(x) for x in out) < max_rows:
        batch=klines(symbol,interval,1000,cur,end_ms)
        if batch.empty: break
        out.append(batch)
        nxt=int(batch.open_time.iloc[-1].timestamp()*1000)+step
        if nxt<=cur: break
        cur=nxt
        if len(batch)<1000: break
    return pd.concat(out,ignore_index=True).drop_duplicates("open_time") if out else pd.DataFrame()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean(); al=l.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))
def atr(df,n=14):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def add_indicators(df):
    x=df.copy()
    x["ema55"]=ema(x.close,55); x["ema200"]=ema(x.close,200)
    x["rsi"]=rsi(x.close); x["atr"]=atr(x)
    x["vol_ma"]=x.volume.rolling(20).mean()
    x["vol_rel"]=x.volume/x.vol_ma
    # Pivots are confirmed two candles later: avoid using future candles in historical signals.
    ph=(x.high>x.high.shift(1))&(x.high>x.high.shift(2))&(x.high>=x.high.shift(-1))&(x.high>=x.high.shift(-2))
    pl=(x.low<x.low.shift(1))&(x.low<x.low.shift(2))&(x.low<=x.low.shift(-1))&(x.low<=x.low.shift(-2))
    x["ph"]=ph.shift(2).fillna(False).astype(bool)
    x["pl"]=pl.shift(2).fillna(False).astype(bool)
    # Keep the original pivot price on the later confirmation candle.
    x["pivot_high"]=x.high.shift(2).where(x["ph"])
    x["pivot_low"]=x.low.shift(2).where(x["pl"])
    return x

def structure_score(x):
    highs=x.loc[x.ph,"pivot_high"].tail(2).values
    lows=x.loc[x.pl,"pivot_low"].tail(2).values
    hs="—"; ls="—"
    if len(highs)==2: hs="HH" if highs[1]>highs[0] else "LH"
    if len(lows)==2: ls="HL" if lows[1]>lows[0] else "LL"
    if hs=="HH" and ls=="HL": return 2,hs,ls
    if hs=="LH" and ls=="LL": return -2,hs,ls
    return 0,hs,ls

def technical_score(df4, df1):
    a=df4.iloc[-1]; b=df1.iloc[-1]
    s4=2 if a.close>a.ema55>a.ema200 and a.ema55>df4.ema55.iloc[-4] else (1 if a.close>a.ema55 else (-2 if a.close<a.ema55<a.ema200 else -1))
    s1=1 if b.close>b.ema55 and b.ema55>df1.ema55.iloc[-4] else (-1 if b.close<b.ema55 and b.ema55<df1.ema55.iloc[-4] else 0)
    mom=1 if 50<=b.rsi<=65 and b.rsi>df1.rsi.iloc[-3] else (-1 if b.rsi<45 else 0)
    vol=1 if b.vol_rel>=1.2 and b.close>b.open else (-1 if b.vol_rel>=1.2 and b.close<b.open else 0)
    st,hs,ls=structure_score(df1)
    return s4,s1,mom,vol,st,hs,ls,s4+s1+mom+vol+st

@st.cache_data(ttl=30)
def current_data(symbol):
    d1=add_indicators(klines(symbol,"1h",500))
    d4=add_indicators(klines(symbol,"4h",500))
    return d1,d4

def derivatives(symbol):
    oi=get_json(FUT+"/futures/data/openInterestHist",{"symbol":symbol,"period":"1h","limit":30})
    funding=get_json(FUT+"/fapi/v1/fundingRate",{"symbol":symbol,"limit":10})
    ticker=get_json(FUT+"/fapi/v1/premiumIndex",{"symbol":symbol})
    return oi,funding,ticker

def derivative_snapshot(symbol):
    try:
        oi, funding, ticker=derivatives(symbol)
        oi_now=float(oi[-1]["sumOpenInterestValue"])
        oi_prev=float(oi[-5]["sumOpenInterestValue"]) if len(oi)>=5 else oi_now
        oi_change=(oi_now/oi_prev-1)*100 if oi_prev else 0
        fr=float(ticker["lastFundingRate"])*100
        # Positive funding supports long crowding; negative supports short crowding.
        if oi_change>2 and fr>0.03: ds=-1
        elif oi_change>2 and fr<-0.03: ds=1
        elif abs(fr)<0.03: ds=0
        else: ds=0
        return oi_now,oi_change,fr,ds
    except Exception as e:
        return None,None,None,0

def plan(df, score, live_price=None):
    p=float(live_price if live_price is not None else df.close.iloc[-1]); A=float(df.atr.iloc[-1])
    if score>=6:
        sw=df.loc[df.pl,"low"].tail(2)
        sl=float(max(sw.iloc[-1] if len(sw) else p-A,p-A))
        tp=p+2*(p-sl)
        return "LONG",p,sl,tp,2.0
    if score<=-6:
        sw=df.loc[df.ph,"high"].tail(2)
        sl=float(min(sw.iloc[-1] if len(sw) else p+A,p+A))
        tp=p-2*(sl-p)
        return "SHORT",p,sl,tp,2.0
    return "WAIT",None,None,None,None

def backtest(df, threshold=6, fee_bps=6, slippage_bps=2, risk_pct=0.5, max_hold=48, start_index=220):
    """Diagnostic 1H backtest; one position at a time, next-open entry, ATR stop/target and trading costs."""
    if df.empty or len(df) < 300:
        return {"Trades": 0, "Win rate %": 0.0, "Profit factor": 0.0, "Expectancy R": 0.0,
                "Net R": 0.0, "Max DD %": 0.0, "Net return %": 0.0}
    x=add_indicators(df).reset_index(drop=True)
    equity=1.0; peak=1.0; maxdd=0.0
    trades=0; wins=0; gross_profit=0.0; gross_loss=0.0; total_r=0.0; outcomes=[]
    cost_per_side=(float(fee_bps)+float(slippage_bps))/10000.0
    i=max(220,int(start_index))
    while i < len(x)-2:
        a=x.iloc[i]
        if not np.isfinite(a.atr) or a.atr <= 0 or not np.isfinite(a.ema200) or not np.isfinite(a.rsi):
            i+=1; continue
        trend=2 if a.close>a.ema55>a.ema200 and a.ema55>x.ema55.iloc[i-4] else (1 if a.close>a.ema55 else (-2 if a.close<a.ema55<a.ema200 else -1))
        conf=1 if a.close>a.ema55 and a.ema55>x.ema55.iloc[i-4] else (-1 if a.close<a.ema55 and a.ema55<x.ema55.iloc[i-4] else 0)
        mom=1 if 50<=a.rsi<=65 and a.rsi>x.rsi.iloc[i-3] else (-1 if a.rsi<45 else 0)
        vol=1 if a.vol_rel>=1.2 and a.close>a.open else (-1 if a.vol_rel>=1.2 and a.close<a.open else 0)
        q=x.iloc[max(0,i-60):i+1]
        stc,_,_=structure_score(q)
        score=trend+conf+mom+vol+stc
        if abs(score)<threshold:
            i+=1; continue
        direction=1 if score>=threshold else -1
        entry=float(x.open.iloc[i+1])
        risk=float(a.atr)
        stop=entry-direction*risk
        target=entry+direction*2*risk
        exit_price=None; exit_idx=None
        last=min(i+1+int(max_hold),len(x)-1)
        for j in range(i+1,last+1):
            bar=x.iloc[j]
            # Conservative assumption: if stop and target are both touched, count the stop first.
            if direction==1:
                if bar.low<=stop: exit_price=stop; exit_idx=j; break
                if bar.high>=target: exit_price=target; exit_idx=j; break
            else:
                if bar.high>=stop: exit_price=stop; exit_idx=j; break
                if bar.low<=target: exit_price=target; exit_idx=j; break
        if exit_price is None:
            exit_idx=last
            exit_price=float(x.close.iloc[exit_idx])
        gross_r=direction*(exit_price-entry)/risk
        costs_r=(2*cost_per_side*entry)/risk
        net_r=gross_r-costs_r
        trades+=1; outcomes.append(net_r); total_r+=net_r
        if net_r>0: wins+=1; gross_profit+=net_r
        else: gross_loss+=abs(net_r)
        equity*=max(0.0,1.0+(float(risk_pct)/100.0)*net_r)
        peak=max(peak,equity)
        if peak>0: maxdd=max(maxdd,(peak-equity)/peak)
        i=exit_idx+1
    expectancy=float(np.mean(outcomes)) if outcomes else 0.0
    return {"Trades":trades,
            "Win rate %":(wins/trades*100.0 if trades else 0.0),
            "Profit factor":(gross_profit/gross_loss if gross_loss>0 else (float("inf") if gross_profit>0 else 0.0)),
            "Expectancy R":expectancy,
            "Net R":total_r,
            "Max DD %":maxdd*100.0,
            "Net return %":(equity-1.0)*100.0}



def backtest_diagnostic(df, threshold=7, filter_mode="Base", fee_bps=6, slippage_bps=2, risk_pct=0.5, max_hold=48, start_index=220, return_trades=False):
    """Diagnostic backtest with optional causal filters and a per-trade audit log."""
    empty = {"Trades":0,"Win rate %":0.0,"Profit factor":0.0,"Expectancy R":0.0,"Net R":0.0,"Max DD %":0.0,"Net return %":0.0}
    if df.empty or len(df)<300:
        return (empty, pd.DataFrame()) if return_trades else empty
    x=add_indicators(df).reset_index(drop=True)
    equity=1.0; peak=1.0; maxdd=0.0; wins=0; gp=0.0; gl=0.0; total_r=0.0; outcomes=[]; logs=[]
    cost_side=(float(fee_bps)+float(slippage_bps))/10000.0
    i=max(220,int(start_index))
    while i < len(x)-2:
        a=x.iloc[i]
        if not np.isfinite(a.atr) or a.atr<=0 or not np.isfinite(a.ema200) or not np.isfinite(a.rsi):
            i+=1; continue
        trend=2 if a.close>a.ema55>a.ema200 and a.ema55>x.ema55.iloc[i-4] else (1 if a.close>a.ema55 else (-2 if a.close<a.ema55<a.ema200 else -1))
        conf=1 if a.close>a.ema55 and a.ema55>x.ema55.iloc[i-4] else (-1 if a.close<a.ema55 and a.ema55<x.ema55.iloc[i-4] else 0)
        mom=1 if 50<=a.rsi<=65 and a.rsi>x.rsi.iloc[i-3] else (-1 if a.rsi<45 else 0)
        vol=1 if a.vol_rel>=1.2 and a.close>a.open else (-1 if a.vol_rel>=1.2 and a.close<a.open else 0)
        stc,_,_=structure_score(x.iloc[max(0,i-60):i+1])
        score=trend+conf+mom+vol+stc
        if abs(score)<threshold:
            i+=1; continue
        direction=1 if score>=threshold else -1
        # Each filter is defined using information available at signal time only.
        if filter_mode=="EMA trend" and not ((direction==1 and a.close>a.ema200) or (direction==-1 and a.close<a.ema200)):
            i+=1; continue
        if filter_mode=="Momentum" and not ((direction==1 and 50<=a.rsi<=65 and a.rsi>x.rsi.iloc[i-3]) or (direction==-1 and a.rsi<45)):
            i+=1; continue
        if filter_mode=="Volume" and not (np.isfinite(a.vol_rel) and a.vol_rel>=1.2):
            i+=1; continue
        if filter_mode=="Structure" and not ((direction==1 and stc>0) or (direction==-1 and stc<0)):
            i+=1; continue
        entry_idx=i+1; entry=float(x.open.iloc[entry_idx]); risk=float(a.atr)
        stop=entry-direction*risk; target=entry+direction*2*risk
        exit_price=None; exit_idx=None; reason="Time exit"
        last=min(entry_idx+int(max_hold),len(x)-1)
        for j in range(entry_idx,last+1):
            bar=x.iloc[j]
            if direction==1:
                if bar.low<=stop: exit_price=stop; exit_idx=j; reason="Stop"; break
                if bar.high>=target: exit_price=target; exit_idx=j; reason="Target"; break
            else:
                if bar.high>=stop: exit_price=stop; exit_idx=j; reason="Stop"; break
                if bar.low<=target: exit_price=target; exit_idx=j; reason="Target"; break
        if exit_price is None:
            exit_idx=last; exit_price=float(x.close.iloc[exit_idx])
        gross_r=direction*(exit_price-entry)/risk
        costs_r=(2*cost_side*entry)/risk
        net_r=gross_r-costs_r
        outcomes.append(net_r); total_r+=net_r
        if net_r>0: wins+=1; gp+=net_r
        else: gl+=abs(net_r)
        equity*=max(0.0,1.0+(float(risk_pct)/100.0)*net_r)
        peak=max(peak,equity)
        if peak>0: maxdd=max(maxdd,(peak-equity)/peak)
        logs.append({"Entrada UTC":x.open_time.iloc[entry_idx],"Salida UTC":x.open_time.iloc[exit_idx],
                     "Dirección":"LONG" if direction==1 else "SHORT","Filtro":filter_mode,"Score":score,
                     "Entrada":entry,"Salida":exit_price,"Stop inicial":stop,"Objetivo inicial":target,
                     "Motivo salida":reason,"R bruto":gross_r,"Costes (R)":costs_r,"R neto":net_r,
                     "Duración (h)":exit_idx-entry_idx})
        i=exit_idx+1
    summary={"Trades":len(outcomes),"Win rate %":wins/len(outcomes)*100 if outcomes else 0.0,
             "Profit factor":gp/gl if gl>0 else (float("inf") if gp>0 else 0.0),
             "Expectancy R":float(np.mean(outcomes)) if outcomes else 0.0,"Net R":total_r,
             "Max DD %":maxdd*100,"Net return %":(equity-1)*100}
    return (summary,pd.DataFrame(logs)) if return_trades else summary

st.title("₿ BTC AI Trader")
st.caption("V1.5  ·  Terminal de análisis cuantitativo  ·  Datos públicos de Binance  ·  Sin conexión a cuentas ni ejecución de órdenes")
st.info("Modo experimental: las señales son heurísticas. El backtest incorpora costes estimados y una prueba cronológica fuera de muestra, pero no demuestra rentabilidad futura.", icon="🧪")

symbol=st.sidebar.selectbox("Símbolo",["BTCUSDT"])
if st.sidebar.button("Actualizar ahora"):
    st.cache_data.clear()

@st.fragment(run_every="30s")
def live_dashboard():
  try:
    live_ticker=get_json(SPOT+"/api/v3/ticker/price", {"symbol":symbol})
    live_price=float(live_ticker["price"])
    d1_raw,d4_raw=current_data(symbol)
    # Exclude the currently forming candle from signal calculations.
    d1=d1_raw.iloc[:-1].copy()
    d4=d4_raw.iloc[:-1].copy()
    s4,s1,mom,vol,stc,hs,ls,score=technical_score(d4,d1)
    oi_now,oi_change,fr,ds=derivative_snapshot(symbol)
    total=max(-8,min(8,score+ds))
    sig,entry,stop,tp,rr=plan(d1,total,live_price)
    if sig!="WAIT" and (rr is None or rr<2): sig="WAIT"
    c1,c2,c3,c4=st.columns(4)
    c1.metric("BTC en tiempo real",f"${live_price:,.2f}", delta=f"{(live_price/float(d1.close.iloc[-2])-1)*100:+.2f}% vs última vela cerrada")
    c2.metric("Score técnico",f"{score:+d}/8")
    c3.metric("Derivados",f"{ds:+d}")
    c4.metric("Señal",sig)
    st.caption("Última actualización de esta lectura: "+datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S %Z"))
    st.divider()
    a,b,c,d,e=st.columns(5)
    a.metric("EMA55 4H",f"${d4.ema55.iloc[-1]:,.0f}")
    b.metric("EMA200 4H",f"${d4.ema200.iloc[-1]:,.0f}")
    c.metric("RSI 1H",f"{d1.rsi.iloc[-1]:.1f}")
    d.metric("Vol. relativo",f"{d1.vol_rel.iloc[-1]:.2f}x")
    e.metric("ATR 1H",f"${d1.atr.iloc[-1]:,.0f}")
    st.subheader("Setup")
    st.write(f"**4H:** {s4:+d} · **1H:** {s1:+d} · **Momentum:** {mom:+d} · **Volumen:** {vol:+d} · **Estructura:** {stc:+d} ({hs}/{ls})")
    if oi_now is not None:
        st.write(f"**Open Interest:** ${oi_now:,.0f} · cambio ~5h: {oi_change:+.2f}% · **Funding:** {fr:+.4f}% · ajuste derivados: {ds:+d}")
    else:
        st.warning("No se pudieron obtener datos de derivados; el score continúa sin ese componente.")
    if sig!="WAIT":
        cols=st.columns(4)
        cols[0].metric("Entrada",f"${entry:,.0f}")
        cols[1].metric("Stop",f"${stop:,.0f}")
        cols[2].metric("TP",f"${tp:,.0f}")
        cols[3].metric("R:R",f"{rr:.2f}")
    else:
        st.info("WAIT — no hay una oportunidad que cumpla todos los filtros de la V1.4.")
    st.subheader("Precio BTC — 1H")
    chart=d1.set_index("open_time")[["close","ema55","ema200"]].tail(240)
    st.line_chart(chart)
    st.subheader("Validación cuantitativa · V1.4")
    st.write("Validación cronológica: el tramo inicial sirve para comparar umbrales; el último 30% se reserva como prueba fuera de muestra (OOS). El umbral se selecciona solo con el tramo de desarrollo, nunca con los resultados OOS. Modelo 1H simplificado; no replica exactamente la señal en vivo.")
    b1,b2,b3,b4=st.columns(4)
    fee_bps=b1.number_input("Comisión por lado (pb)",min_value=0.0,max_value=100.0,value=6.0,step=1.0,key="bt_fee_bps")
    slippage_bps=b2.number_input("Deslizamiento por lado (pb)",min_value=0.0,max_value=100.0,value=2.0,step=1.0,key="bt_slippage_bps")
    risk_pct=b3.number_input("Riesgo por operación (%)",min_value=0.1,max_value=5.0,value=0.5,step=0.1,key="bt_risk_pct")
    max_hold=b4.number_input("Máx. duración (velas 1H)",min_value=1,max_value=240,value=48,step=1,key="bt_max_hold")
    if st.button("Ejecutar validación V1.4",key="run_backtest_v14"):
        try:
            with st.spinner("Descargando histórico y calculando…"):
                end=int(datetime.now(timezone.utc).timestamp()*1000)
                start=int((datetime.now(timezone.utc)-timedelta(days=365*5)).timestamp()*1000)
                hist=paginate_klines(symbol,"1h",start,end,12000)
                if hist.empty or len(hist)<800:
                    st.session_state["bt_error"]="Histórico insuficiente para una validación cronológica fiable. Prueba de nuevo más tarde."
                    st.session_state.pop("bt_results",None)
                    st.session_state.pop("bt_oos",None)
                else:
                    cut=int(len(hist)*0.70)
                    development=hist.iloc[:cut].copy()
                    # Include a 300-candle warm-up before OOS, but start trading exactly at the split.
                    warm=max(0,cut-300)
                    test=hist.iloc[warm:].copy()
                    start_test=cut-warm
                    rows=[]
                    for th in [5,6,7]:
                        row=backtest(development,th,fee_bps,slippage_bps,risk_pct,int(max_hold))
                        row["Umbral"]=th
                        rows.append(row)
                    dev_table=pd.DataFrame(rows).set_index("Umbral")
                    candidates=dev_table.replace([np.inf,-np.inf],np.nan)
                    # Choose using development expectancy only; tie-break on profit factor.
                    chosen=int(candidates.sort_values(["Expectancy R","Profit factor"],ascending=False).index[0])
                    oos=backtest(test,chosen,fee_bps,slippage_bps,risk_pct,int(max_hold),start_index=start_test)
                    st.session_state["bt_results"]=dev_table
                    st.session_state["bt_oos"]=pd.DataFrame([{"Umbral seleccionado (solo desarrollo)":chosen,**oos}])
                    st.session_state["bt_split"]=cut
                    st.session_state["bt_history_count"]=len(hist)
                    st.session_state.pop("bt_error",None)
        except Exception as e:
            st.session_state["bt_error"]=f"{type(e).__name__}: {e}"
            st.session_state.pop("bt_results",None)
    if st.session_state.get("bt_error"):
        st.error("El backtest no ha podido terminar: "+st.session_state["bt_error"])
    if "bt_results" in st.session_state:
        bt=st.session_state["bt_results"]
        split=st.session_state.get("bt_split",0)
        st.caption(f"Histórico analizado: {st.session_state.get('bt_history_count',0):,} velas 1H · desarrollo: {split:,} velas (70%) · prueba OOS: {st.session_state.get('bt_history_count',0)-split:,} velas (30%). El histórico está limitado a 12.000 velas, aproximadamente 16 meses.")
        st.markdown("#### 1 · Desarrollo — comparación de umbrales")
        st.dataframe(bt.style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
        if "bt_oos" in st.session_state:
            st.markdown("#### 2 · Prueba fuera de muestra — umbral elegido en desarrollo")
            oos_table=st.session_state["bt_oos"]
            st.dataframe(oos_table.style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
            exp=float(oos_table["Expectancy R"].iloc[0])
            if exp>0:
                st.warning("La expectativa OOS es positiva en esta muestra, pero aún requiere más periodos y pruebas de robustez; no es garantía de rentabilidad.")
            else:
                st.error("La expectativa OOS es negativa o nula: el sistema no supera esta prueba fuera de muestra. No usar para operar en real.")
        export=bt.reset_index().assign(Segmento="Desarrollo")
        if "bt_oos" in st.session_state:
            export_oos=st.session_state["bt_oos"].rename(columns={"Umbral seleccionado (solo desarrollo)":"Umbral"})
            export_oos["Segmento"]="Fuera de muestra"
            export=pd.concat([export,export_oos],ignore_index=True,sort=False)
        st.download_button("Descargar informe CSV",export.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_validation_v1_4.csv",mime="text/csv",key="download_backtest_v14")

    st.divider()
    st.subheader("Laboratorio de operaciones · V1.5")
    st.write("Auditoría por operación y comparación de filtros. Se usa un umbral fijo de 7 para comparar los filtros de forma homogénea. El tramo OOS queda reservado para evaluar el filtro seleccionado en desarrollo; como se prueban varias alternativas, el resultado sigue siendo exploratorio.")
    if st.button("Ejecutar diagnóstico V1.5",key="run_diagnostic_v15"):
        try:
            with st.spinner("Analizando operaciones y filtros…"):
                end2=int(datetime.now(timezone.utc).timestamp()*1000)
                start2=int((datetime.now(timezone.utc)-timedelta(days=365*5)).timestamp()*1000)
                hist15=paginate_klines(symbol,"1h",start2,end2,12000)
                if hist15.empty or len(hist15)<800:
                    st.session_state["diag_error"]="No hay suficiente histórico para separar desarrollo y prueba."
                    st.session_state.pop("diag_compare",None); st.session_state.pop("diag_oos",None); st.session_state.pop("diag_trades",None)
                else:
                    cut15=int(len(hist15)*0.70)
                    dev15=hist15.iloc[:cut15].copy()
                    warm15=max(0,cut15-300)
                    test15=hist15.iloc[warm15:].copy()
                    start_oos15=cut15-warm15
                    modes=["Base","EMA trend","Momentum","Volume","Structure"]
                    comparison=[]
                    for mode in modes:
                        met=backtest_diagnostic(dev15,7,mode,fee_bps,slippage_bps,risk_pct,int(max_hold))
                        comparison.append({"Filtro":mode,**met})
                    comp15=pd.DataFrame(comparison)
                    ranked=comp15.replace([np.inf,-np.inf],np.nan).sort_values(["Expectancy R","Profit factor"],ascending=False)
                    chosen15=str(ranked.iloc[0]["Filtro"])
                    oos15, trades15=backtest_diagnostic(test15,7,chosen15,fee_bps,slippage_bps,risk_pct,int(max_hold),start_index=start_oos15,return_trades=True)
                    # Also retain all OOS trades for a side-by-side long/short and exit-reason diagnosis.
                    st.session_state["diag_compare"]=comp15
                    st.session_state["diag_oos"]=pd.DataFrame([{"Filtro elegido en desarrollo":chosen15,**oos15}])
                    st.session_state["diag_trades"]=trades15
                    st.session_state["diag_count"]=len(hist15)
                    st.session_state["diag_cut"]=cut15
                    st.session_state.pop("diag_error",None)
        except Exception as e:
            st.session_state["diag_error"]=f"{type(e).__name__}: {e}"
            st.session_state.pop("diag_compare",None); st.session_state.pop("diag_oos",None); st.session_state.pop("diag_trades",None)
    if st.session_state.get("diag_error"):
        st.error("No se pudo completar el diagnóstico: "+st.session_state["diag_error"])
    if "diag_compare" in st.session_state:
        comp15=st.session_state["diag_compare"]
        st.caption(f"Histórico: {st.session_state.get('diag_count',0):,} velas 1H · desarrollo: {st.session_state.get('diag_cut',0):,} · OOS: {st.session_state.get('diag_count',0)-st.session_state.get('diag_cut',0):,}. Umbral fijo = 7.")
        st.markdown("#### Comparación de filtros — desarrollo")
        st.dataframe(comp15.style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
        st.markdown("#### Evaluación fuera de muestra")
        st.dataframe(st.session_state["diag_oos"].style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
        trades15=st.session_state["diag_trades"]
        if not trades15.empty:
            st.markdown("#### Diagnóstico de operaciones OOS")
            t1,t2=st.columns(2)
            t1.markdown("**Por dirección**")
            st1=trades15.groupby("Dirección").agg(Operaciones=("R neto","count"),Acierto_pct=("R neto",lambda v:(v>0).mean()*100),Expectativa_R=("R neto","mean"),R_neto=("R neto","sum")).reset_index()
            t1.dataframe(st1.style.format({"Acierto_pct":"{:.1f}","Expectativa_R":"{:.3f}","R_neto":"{:.2f}"}),use_container_width=True)
            t2.markdown("**Por salida**")
            st2=trades15.groupby("Motivo salida").agg(Operaciones=("R neto","count"),Expectativa_R=("R neto","mean"),R_neto=("R neto","sum")).reset_index()
            t2.dataframe(st2.style.format({"Expectativa_R":"{:.3f}","R_neto":"{:.2f}"}),use_container_width=True)
            st.markdown("#### Registro detallado")
            st.dataframe(trades15,use_container_width=True)
            st.download_button("Descargar operaciones OOS CSV",trades15.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_trades_v1_5.csv",mime="text/csv",key="download_trades_v15")
        else:
            st.warning("No se generaron operaciones en el tramo OOS para este filtro.")

  except Exception as e:
    st.error("No se pudieron cargar los datos. Comprueba la conexión o vuelve a actualizar.")
    st.caption(str(e))

live_dashboard()
