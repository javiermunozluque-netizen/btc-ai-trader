# BUILD SYNC: 2026-10-07 V1.9.2

import streamlit as st
import pandas as pd
import numpy as np
import requests, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="BTC AI Trader V1.9.2", page_icon="₿", layout="wide")

st.markdown("""<style>
:root{--btc-accent:#f7931a}
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1600px}
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(247,147,26,.075),transparent 34%),radial-gradient(ellipse at 95% 8%,rgba(85,119,255,.07),transparent 30%)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(127,140,160,.12),rgba(127,140,160,.035));border:1px solid rgba(127,140,160,.24);padding:17px 19px;border-radius:16px;box-shadow:0 5px 20px rgba(0,0,0,.035);min-height:108px}
[data-testid="stMetricLabel"]{font-size:.78rem;letter-spacing:.045em;text-transform:uppercase;opacity:.78}
[data-testid="stMetricValue"]{font-weight:750;letter-spacing:-.035em}
section[data-testid="stSidebar"]{border-right:1px solid rgba(127,140,160,.2)}
.stTabs [data-baseweb="tab-list"]{gap:7px;border-bottom:1px solid rgba(127,140,160,.22)}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:11px 16px;font-weight:600}
.stTabs [aria-selected="true"]{border-bottom:2px solid var(--btc-accent)}
div[data-testid="stAlert"]{border-radius:13px;border:1px solid rgba(127,140,160,.2)}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;font-weight:650;min-height:2.65rem}
div.stButton>button[kind="primary"]{background:var(--btc-accent);border-color:var(--btc-accent);color:#171717}
div[data-testid="stDataFrame"]{border:1px solid rgba(127,140,160,.22);border-radius:12px;overflow:hidden}
h1{font-weight:800;letter-spacing:-.055em} h2,h3{letter-spacing:-.03em}
hr{margin:1.35rem 0;border-color:rgba(127,140,160,.2)}
@media(max-width:700px){.block-container{padding-top:.7rem;padding-left:1rem;padding-right:1rem}[data-testid="stMetric"]{padding:12px;min-height:94px}[data-testid="stMetricValue"]{font-size:1.35rem}}
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
    out=[]
    step={"1h":3600000,"4h":14400000}[interval]
    # Use the recent edge of the requested window, not old candles first.
    cur=max(int(start_ms),int(end_ms)-int(max_rows)*step)
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

@st.cache_data(show_spinner=False)
def prepare_diagnostic(df):
    x=add_indicators(df).reset_index(drop=True).copy()
    stc_vals=np.zeros(len(x),dtype=int)
    for k in range(220,len(x)):
        stc_vals[k]=structure_score(x.iloc[max(0,k-60):k+1])[0]
    x["structure_score"]=stc_vals
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
    x=prepare_diagnostic(df)
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



def backtest_diagnostic(df, threshold=7, filter_mode="Base", fee_bps=6, slippage_bps=2, risk_pct=0.5, max_hold=48, start_index=220, return_trades=False, direction_mode="Both"):
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
        stc=int(x.structure_score.iloc[i])
        score=trend+conf+mom+vol+stc
        if abs(score)<threshold:
            i+=1; continue
        direction=1 if score>=threshold else -1
        # Direction filters are evaluated at signal time; they do not change the score.
        if direction_mode=="LONG only" and direction!=1:
            i+=1; continue
        if direction_mode=="SHORT only" and direction!=-1:
            i+=1; continue
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
        fee_cost_side=float(fee_bps)/10000.0*entry
        slip_cost_side=float(slippage_bps)/10000.0*entry
        fee_total=2*fee_cost_side
        slip_total=2*slip_cost_side
        total_cost_usdt=fee_total+slip_total
        ambiguous_bar=(direction==1 and x.low.iloc[exit_idx]<=stop and x.high.iloc[exit_idx]>=target) or (direction==-1 and x.high.iloc[exit_idx]>=stop and x.low.iloc[exit_idx]<=target)
        logs.append({"Entrada UTC":x.open_time.iloc[entry_idx],"Salida UTC":x.open_time.iloc[exit_idx],
                     "Dirección":"LONG" if direction==1 else "SHORT","Filtro":filter_mode,"Score":score,
                     "Entrada":entry,"Salida":exit_price,"Stop inicial":stop,"Objetivo inicial":target,
                     "Motivo salida":reason,"R bruto":gross_r,"Comisión (USDT)":fee_total,
                     "Deslizamiento (USDT)":slip_total,"Costes (USDT)":total_cost_usdt,
                     "Costes (R)":costs_r,"R neto":net_r,"Duración (h)":exit_idx-entry_idx,
                     "Misma vela":exit_idx==entry_idx,"Vela ambigua":bool(ambiguous_bar)})
        i=exit_idx+1
    summary={"Trades":len(outcomes),"Win rate %":wins/len(outcomes)*100 if outcomes else 0.0,
             "Profit factor":gp/gl if gl>0 else (float("inf") if gp>0 else 0.0),
             "Expectancy R":float(np.mean(outcomes)) if outcomes else 0.0,"Net R":total_r,
             "Max DD %":maxdd*100,"Net return %":(equity-1)*100}
    return (summary,pd.DataFrame(logs)) if return_trades else summary

st.markdown("""
<div style="padding:22px 24px;margin:2px 0 16px;border:1px solid rgba(127,140,160,.22);border-radius:18px;background:linear-gradient(115deg,rgba(247,147,26,.12),rgba(127,140,160,.035) 52%,rgba(85,119,255,.08));">
  <div style="font-size:.76rem;font-weight:750;letter-spacing:.14em;text-transform:uppercase;opacity:.72;margin-bottom:7px">QUANT RESEARCH · BTC / USDT</div>
  <div style="font-size:clamp(1.8rem,4vw,2.7rem);font-weight:850;letter-spacing:-.055em;line-height:1.08">₿ BTC AI Trader <span style="color:#f7931a">/ V1.9.2</span></div>
  <div style="margin-top:9px;font-size:.96rem;opacity:.82">Market intelligence · Backtest audit · Robust diagnostics</div>
  <div style="display:inline-block;margin-top:15px;padding:5px 10px;border:1px solid rgba(127,140,160,.28);border-radius:99px;font-size:.75rem;font-weight:650">● DATOS PÚBLICOS · SOLO ANÁLISIS · SIN EJECUCIÓN DE ÓRDENES</div>
</div>
""",unsafe_allow_html=True)
st.caption("Terminal experimental de análisis cuantitativo. No conecta cuentas de exchange ni utiliza claves privadas.")
st.info("Modo experimental: las señales son heurísticas. El backtest incorpora costes estimados y una prueba cronológica fuera de muestra, pero no demuestra rentabilidad futura.", icon="🧪")

symbol=st.sidebar.selectbox("Símbolo",["BTCUSDT"])
if st.sidebar.button("Actualizar ahora"):
    st.cache_data.clear()

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
        st.info("WAIT — no hay una oportunidad que cumpla los criterios actuales.")
    st.subheader("Estructura de mercado · BTC/USDT")
    st.caption("Velas japonesas con medias móviles y volumen. Usa el zoom, desplázate por el gráfico y selecciona el rango temporal.")
    chart_df=d1.tail(240).copy()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.045,row_heights=[0.76,0.24],subplot_titles=("Precio · velas 1H","Volumen"))
    fig.add_trace(go.Candlestick(x=chart_df.open_time,open=chart_df.open,high=chart_df.high,low=chart_df.low,close=chart_df.close,name="BTC/USDT",increasing_line_color="#16a085",decreasing_line_color="#e05a5a"),row=1,col=1)
    fig.add_trace(go.Scatter(x=chart_df.open_time,y=chart_df.ema55,mode="lines",name="EMA 55",line=dict(color="#f7931a",width=1.7)),row=1,col=1)
    fig.add_trace(go.Scatter(x=chart_df.open_time,y=chart_df.ema200,mode="lines",name="EMA 200",line=dict(color="#748ffc",width=1.7)),row=1,col=1)
    vol_colors=["#16a085" if cl>=op else "#e05a5a" for cl,op in zip(chart_df.close,chart_df.open)]
    fig.add_trace(go.Bar(x=chart_df.open_time,y=chart_df.volume,name="Volumen",marker_color=vol_colors,showlegend=False),row=2,col=1)
    fig.update_layout(height=610,template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white",margin=dict(l=8,r=8,t=45,b=8),hovermode="x unified",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="USDT",row=1,col=1,side="right",gridcolor="rgba(127,140,160,.18)")
    fig.update_yaxes(title_text="Volumen",row=2,col=1,side="right",gridcolor="rgba(127,140,160,.18)")
    fig.update_xaxes(showgrid=False,row=1,col=1)
    fig.update_xaxes(showgrid=False,row=2,col=1)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    st.subheader("Validación cuantitativa · V1.9.2")
    st.caption("Los análisis se ejecutan solo al pulsar el botón. V1.9 añade una auditoría de ejecución: causalidad, velas ambiguas, salidas en la misma vela y sensibilidad a costes.")
    st.write("Validación cronológica: comparación de umbrales en desarrollo y evaluación en el 30% final fuera de muestra (OOS). Modelo 1H simplificado; no replica exactamente la señal multi-timeframe en vivo.")
    b1,b2,b3,b4=st.columns(4)
    fee_bps=b1.number_input("Comisión por lado (pb)",min_value=0.0,max_value=100.0,value=6.0,step=1.0,key="bt_fee_bps")
    slippage_bps=b2.number_input("Deslizamiento por lado (pb)",min_value=0.0,max_value=100.0,value=2.0,step=1.0,key="bt_slippage_bps")
    risk_pct=b3.number_input("Riesgo por operación (%)",min_value=0.1,max_value=5.0,value=0.5,step=0.1,key="bt_risk_pct")
    max_hold=b4.number_input("Máx. duración (velas 1H)",min_value=1,max_value=240,value=48,step=1,key="bt_max_hold")
    if st.button("Ejecutar validación V1.9",key="run_backtest_v191"):
        try:
            with st.spinner("Descargando hasta 12.000 velas recientes y calculando métricas. Puede tardar un poco…"):
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
                    # Keep the full series for causal indicator warm-up; trade only after the split.
                    test=hist.copy()
                    start_test=cut
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
        st.caption(f"Histórico analizado: {st.session_state.get('bt_history_count',0):,} velas 1H · desarrollo: {split:,} velas (70%) · prueba OOS: {st.session_state.get('bt_history_count',0)-split:,} velas (30%). Se usan las últimas 12.000 velas disponibles, no las primeras desde la fecha inicial.")
        st.success("Validación completada correctamente.")
        st.markdown("#### 1 · Desarrollo — comparación de umbrales")
        st.caption("Cada fila representa un umbral probado sobre el tramo de desarrollo. El umbral seleccionado se utiliza después, sin recalibrarlo, en OOS.")
        bt_view=bt.reset_index().rename(columns={"Umbral":"Umbral","Win rate %":"Win Rate","Profit factor":"Profit Factor","Expectancy R":"Expectativa R","Net R":"R neto","Max DD %":"Max DD","Net return %":"Rentabilidad"})
        m1,m2,m3,m4,m5=st.columns(5)
        best_row=bt.loc[bt["Expectancy R"].idxmax()]
        m1.metric("Umbral elegido",str(int(bt["Expectancy R"].idxmax())))
        m2.metric("Operaciones",f"{int(best_row['Trades'])}")
        m3.metric("Win Rate",f"{best_row['Win rate %']:.1f}%")
        m4.metric("Expectativa",f"{best_row['Expectancy R']:.3f} R")
        m5.metric("R neto",f"{best_row['Net R']:+.2f} R")
        st.dataframe(bt_view.round({"Win Rate":1,"Profit Factor":2,"Expectativa R":3,"R neto":2,"Max DD":1,"Rentabilidad":1}),use_container_width=True,hide_index=True)
        if "bt_oos" in st.session_state:
            st.markdown("#### 2 · Prueba fuera de muestra — resultado real del modelo")
            oos_table=st.session_state["bt_oos"].copy()
            oos_view=oos_table.rename(columns={"Umbral seleccionado (solo desarrollo)":"Configuración","Win rate %":"Win Rate","Profit factor":"Profit Factor","Expectancy R":"Expectativa R","Net R":"R neto","Max DD %":"Max DD","Net return %":"Rentabilidad"})
            oos_row=oos_table.iloc[0]
            om1,om2,om3,om4,om5=st.columns(5)
            om1.metric("Umbral",str(int(oos_row["Umbral seleccionado (solo desarrollo)"])))
            om2.metric("Operaciones",f"{int(oos_row['Trades'])}")
            om3.metric("Win Rate",f"{oos_row['Win rate %']:.1f}%")
            om4.metric("Expectativa",f"{oos_row['Expectancy R']:.3f} R")
            om5.metric("R neto",f"{oos_row['Net R']:+.2f} R")
            st.dataframe(oos_view.round({"Win Rate":1,"Profit Factor":2,"Expectativa R":3,"R neto":2,"Max DD":1,"Rentabilidad":1}),use_container_width=True,hide_index=True)
            exp=float(oos_row["Expectancy R"])
            if exp>0:
                st.warning("La expectativa OOS es positiva en esta muestra, pero aún requiere más periodos y pruebas de robustez; no es garantía de rentabilidad.")
            else:
                st.error("La expectativa OOS es negativa o nula: el sistema no supera esta prueba fuera de muestra. No usar para operar en real.")
        export=bt.reset_index().assign(Segmento="Desarrollo")
        if "bt_oos" in st.session_state:
            export_oos=st.session_state["bt_oos"].rename(columns={"Umbral seleccionado (solo desarrollo)":"Umbral"})
            export_oos["Segmento"]="Fuera de muestra"
            export=pd.concat([export,export_oos],ignore_index=True,sort=False)
        st.download_button("Descargar informe CSV",export.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_validation_v1_9_2.csv",mime="text/csv",key="download_backtest_v19")

    st.divider()
    st.subheader("Laboratorio cuantitativo · V1.9")
    st.write("Compara filtros técnicos y dirección de operación con un umbral fijo de 7. La selección se hace solo en desarrollo y se evalúa en el 30% final OOS. Para evitar elegir configuraciones con muy pocas operaciones, se priorizan candidatos con al menos 15 operaciones de desarrollo; aun así, el resultado es exploratorio.")
    if st.button("Ejecutar diagnóstico V1.9",key="run_diagnostic_v19"):
        try:
            with st.spinner("Comparando filtros y modos LONG/SHORT sobre el histórico reciente…"):
                end2=int(datetime.now(timezone.utc).timestamp()*1000)
                start2=int((datetime.now(timezone.utc)-timedelta(days=365*5)).timestamp()*1000)
                hist15=paginate_klines(symbol,"1h",start2,end2,12000)
                if hist15.empty or len(hist15)<800:
                    st.session_state["diag_error"]="No hay suficiente histórico para separar desarrollo y prueba."
                    st.session_state.pop("diag_compare",None); st.session_state.pop("diag_oos",None); st.session_state.pop("diag_trades",None)
                else:
                    cut15=int(len(hist15)*0.70)
                    dev15=hist15.iloc[:cut15].copy()
                    test15=hist15.copy()
                    start_oos15=cut15
                    thresholds=[5,6,7,8]
                    modes=["Base","EMA trend","Momentum","Volume","Structure"]
                    directions=["Both","LONG only","SHORT only"]
                    comparison=[]
                    for threshold in thresholds:
                        for mode in modes:
                            for direction_mode in directions:
                                met=backtest_diagnostic(dev15,threshold,mode,fee_bps,slippage_bps,risk_pct,int(max_hold),direction_mode=direction_mode)
                                comparison.append({"Umbral":threshold,"Filtro":mode,"Dirección":direction_mode,**met})
                    comp15=pd.DataFrame(comparison)
                    ranked=comp15.replace([np.inf,-np.inf],np.nan)
                    eligible=ranked[ranked["Trades"]>=15]
                    if eligible.empty:
                        eligible=ranked[ranked["Trades"]>=5]
                    if eligible.empty:
                        eligible=ranked
                    ranked=eligible.sort_values(["Expectancy R","Profit factor"],ascending=False)
                    chosen_threshold=int(ranked.iloc[0]["Umbral"])
                    chosen_filter=str(ranked.iloc[0]["Filtro"])
                    chosen_direction=str(ranked.iloc[0]["Dirección"])
                    chosen15=f"Score {chosen_threshold} · {chosen_filter} · {chosen_direction}"
                    oos15, trades15=backtest_diagnostic(test15,chosen_threshold,chosen_filter,fee_bps,slippage_bps,risk_pct,int(max_hold),start_index=start_oos15,return_trades=True,direction_mode=chosen_direction)
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
        st.caption(f"Histórico: {st.session_state.get('diag_count',0):,} velas 1H · desarrollo: {st.session_state.get('diag_cut',0):,} · OOS: {st.session_state.get('diag_count',0)-st.session_state.get('diag_cut',0):,}. Se comparan 60 combinaciones: 4 umbrales × 5 filtros × 3 direcciones. La selección se hace solo con desarrollo y se valida después en OOS.")
        st.success("Diagnóstico completado correctamente.")
        st.markdown("#### Comparación visual — expectativa por filtro y dirección")
        chart_comp=comp15.copy()
        chart_comp["Configuración"]=chart_comp["Filtro"]+" · "+chart_comp["Dirección"]
        chart_comp=chart_comp.sort_values("Expectancy R",ascending=True)
        comp_fig=go.Figure()
        bar_colors=["#16a085" if v>0 else "#e05a5a" for v in chart_comp["Expectancy R"]]
        comp_fig.add_trace(go.Bar(x=chart_comp["Expectancy R"],y=chart_comp["Configuración"],orientation="h",marker_color=bar_colors,hovertemplate="%{y}<br>Expectativa: %{x:.3f} R/operación<extra></extra>"))
        comp_fig.add_vline(x=0,line_dash="dash",line_color="gray")
        comp_fig.update_layout(height=510,template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white",margin=dict(l=8,r=8,t=15,b=8),xaxis_title="Expectativa neta (R por operación)",yaxis_title="",showlegend=False)
        comp_fig.update_xaxes(gridcolor="rgba(127,140,160,.18)")
        comp_fig.update_yaxes(showgrid=False)
        st.plotly_chart(comp_fig,use_container_width=True,config={"displaylogo":False})
        st.caption("Verde = expectativa media positiva en desarrollo; rojo = negativa. La configuración se elige solo con el tramo de desarrollo y después se evalúa en OOS.")
        st.markdown("#### Tabla detallada — desarrollo")
        st.dataframe(comp15.style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
        st.markdown("#### Evaluación fuera de muestra")
        st.dataframe(st.session_state["diag_oos"].style.format({"Win rate %":"{:.1f}","Profit factor":"{:.2f}","Expectancy R":"{:.3f}","Net R":"{:.2f}","Max DD %":"{:.1f}","Net return %":"{:.1f}"}),use_container_width=True)
        trades15=st.session_state["diag_trades"]
        if not trades15.empty:
            st.markdown("#### Curva de resultados fuera de muestra")
            plot_trades=trades15.sort_values("Salida UTC").copy()
            plot_trades["R acumulado"]=plot_trades["R neto"].cumsum()
            plot_trades["Máximo acumulado"]=plot_trades["R acumulado"].cummax().clip(lower=0)
            plot_trades["Drawdown (R)"]=plot_trades["R acumulado"]-plot_trades["Máximo acumulado"]
            curve=go.Figure()
            curve.add_trace(go.Scatter(x=plot_trades["Salida UTC"],y=plot_trades["R acumulado"],mode="lines+markers",name="R acumulado",line=dict(color="#f7931a",width=2.5),marker=dict(size=5),hovertemplate="%{x}<br>R acumulado: %{y:.2f}<extra></extra>"))
            curve.add_hline(y=0,line_dash="dash",line_color="gray",opacity=.65)
            curve.update_layout(height=330,template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white",margin=dict(l=8,r=8,t=20,b=8),xaxis_title="Fecha de salida",yaxis_title="R acumulado",showlegend=False,hovermode="x unified")
            curve.update_xaxes(showgrid=False)
            curve.update_yaxes(gridcolor="rgba(127,140,160,.18)")
            st.plotly_chart(curve,use_container_width=True,config={"displaylogo":False})
            k1,k2,k3=st.columns(3)
            k1.metric("Operaciones OOS",f"{len(plot_trades)}")
            k2.metric("Resultado acumulado",f"{plot_trades['R neto'].sum():+.2f} R")
            k3.metric("Peor drawdown",f"{plot_trades['Drawdown (R)'].min():.2f} R")
            st.markdown("#### Diagnóstico de operaciones OOS")
            t1,t2=st.columns(2)
            t1.markdown("**Por dirección**")
            st1=trades15.groupby("Dirección").agg(Operaciones=("R neto","count"),Acierto_pct=("R neto",lambda v:(v>0).mean()*100),Expectativa_R=("R neto","mean"),R_neto=("R neto","sum")).reset_index()
            t1.dataframe(st1.style.format({"Acierto_pct":"{:.1f}","Expectativa_R":"{:.3f}","R_neto":"{:.2f}"}),use_container_width=True)
            t2.markdown("**Por salida**")
            st2=trades15.groupby("Motivo salida").agg(Operaciones=("R neto","count"),Expectativa_R=("R neto","mean"),R_neto=("R neto","sum")).reset_index()
            t2.dataframe(st2.style.format({"Expectativa_R":"{:.3f}","R_neto":"{:.2f}"}),use_container_width=True)
            st.markdown("#### Auditoría de ejecución V1.9")
            audit=trades15.copy()
            zero_h=int((audit["Duración (h)"]==0).sum())
            ambiguous=int(audit["Vela ambigua"].sum()) if "Vela ambigua" in audit.columns else 0
            same_bar=int(audit["Misma vela"].sum()) if "Misma vela" in audit.columns else 0
            avg_cost=float(audit["Costes (R)"].mean()) if not audit.empty else 0.0
            total_cost=float(audit["Costes (USDT)"].sum()) if "Costes (USDT)" in audit.columns else 0.0
            a1,a2,a3,a4,a5=st.columns(5)
            a1.metric("Salidas 0h",f"{zero_h} ({zero_h/len(audit)*100:.1f}%)")
            a2.metric("Salidas misma vela",f"{same_bar}")
            a3.metric("Velas ambiguas",f"{ambiguous}")
            a4.metric("Coste medio",f"{avg_cost:.3f} R")
            a5.metric("Coste total",f"{total_cost:,.0f} USDT")
            st.caption("Una salida de 0h no es necesariamente un error: la operación entra en la apertura de una vela y el stop/objetivo puede alcanzarse dentro de esa misma vela. Si una vela toca simultáneamente stop y objetivo, V1.9 aplica stop-first de forma conservadora.")

            st.markdown("#### Sensibilidad a costes")
            st.caption("Mantiene exactamente las mismas entradas y salidas OOS y modifica únicamente la fricción.")
            sens=[]
            for total_bps in [0,4,8,12,16,20]:
                gross=float(audit["R bruto"].sum())
                denom=(audit["Stop inicial"]-audit["Entrada"]).abs()
                sens_cost=float((total_bps/10000.0*audit["Entrada"]/denom).sum())
                sens.append({"Coste total ida+vuelta (pb)":total_bps,"Costes (R)":sens_cost,"R neto":gross-sens_cost,"Expectativa R":(gross-sens_cost)/len(audit)})
            sens_df=pd.DataFrame(sens)
            sens_fig=go.Figure()
            sens_fig.add_trace(go.Scatter(x=sens_df["Coste total ida+vuelta (pb)"],y=sens_df["R neto"],mode="lines+markers",name="R neto",line=dict(width=2.5)))
            sens_fig.add_hline(y=0,line_dash="dash",line_color="gray")
            sens_fig.update_layout(height=300,template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white",margin=dict(l=8,r=8,t=15,b=8),xaxis_title="Coste total ida + vuelta (pb)",yaxis_title="R neto",showlegend=False)
            sens_fig.update_xaxes(gridcolor="rgba(127,140,160,.18)")
            sens_fig.update_yaxes(gridcolor="rgba(127,140,160,.18)")
            st.plotly_chart(sens_fig,use_container_width=True,config={"displaylogo":False})
            st.dataframe(sens_df.style.format({"Costes (R)":"{:.2f}","R neto":"{:.2f}","Expectativa R":"{:.3f}"}),use_container_width=True)

            st.markdown("#### Registro detallado")
            st.dataframe(trades15,use_container_width=True)
            st.download_button("Descargar operaciones OOS CSV",trades15.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_trades_v1_9_2.csv",mime="text/csv",key="download_trades_v192")
        else:
            st.warning("No se generaron operaciones en el tramo OOS para este filtro.")

  except Exception as e:
    st.error("No se pudieron cargar los datos. Comprueba la conexión o vuelve a actualizar.")
    st.caption(str(e))

live_dashboard()
