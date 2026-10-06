
import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="BTC AI Trader V1.2", page_icon="₿", layout="wide")

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
    # Simple confirmed swing structure using 3-bar pivots
    x["ph"]=(x.high>x.high.shift(1))&(x.high>x.high.shift(2))&(x.high>=x.high.shift(-1))&(x.high>=x.high.shift(-2))
    x["pl"]=(x.low<x.low.shift(1))&(x.low<x.low.shift(2))&(x.low<=x.low.shift(-1))&(x.low<=x.low.shift(-2))
    return x

def structure_score(x):
    highs=x.loc[x.ph,"high"].tail(2).values
    lows=x.loc[x.pl,"low"].tail(2).values
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

def backtest(df, threshold=6, fee=0.0006):
    x=add_indicators(df).reset_index(drop=True)
    equity=1.0; peak=1.0; maxdd=0; wins=losses=trades=0; R=0
    i=220
    while i<len(x)-2:
        s=technical_score(x.iloc[:i+1], x.iloc[:i+1])[7] if False else None
        # score on 1H alone: trend + confirmation + momentum + volume + structure
        a=x.iloc[i]
        trend=2 if a.close>a.ema55>a.ema200 and a.ema55>x.ema55.iloc[i-4] else (1 if a.close>a.ema55 else (-2 if a.close<a.ema55<a.ema200 else -1))
        conf=1 if a.close>a.ema55 and a.ema55>x.ema55.iloc[i-4] else (-1 if a.close<a.ema55 and a.ema55<x.ema55.iloc[i-4] else 0)
        mom=1 if 50<=a.rsi<=65 and a.rsi>x.rsi.iloc[i-3] else (-1 if a.rsi<45 else 0)
        vol=1 if a.vol_rel>=1.2 and a.close>a.open else (-1 if a.vol_rel>=1.2 and a.close<a.open else 0)
        # local structure
        q=x.iloc[max(0,i-60):i+1]
        st,_,_=structure_score(q)
        score=trend+conf+mom+vol+st
        if abs(score)<threshold: i+=1; continue
        direction=1 if score>=threshold else -1
        entry=float(x.open.iloc[i+1]); A=float(a.atr); stop=entry-direction*A; target=entry+direction*2*A
        result=None
        for j in range(i+1,min(i+49,len(x))):
            if direction==1:
                if x.low.iloc[j]<=stop: result=-1; break
                if x.high.iloc[j]>=target: result=2; break
            else:
                if x.high.iloc[j]>=stop: result=-1; break
                if x.low.iloc[j]<=target: result=2; break
        if result is None: i+=1; continue
        trades+=1
        net=(result*0.5) - fee*2 # simplified R-equivalent approximation
        R+=result
        if result==2: wins+=1
        else: losses+=1
        equity*=1+net*0.005
        peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak)
        i=j+1
    return trades,wins,losses,(wins/(wins+losses)*100 if wins+losses else 0),R,maxdd

st.title("₿ BTC AI Trader — V1.2")
st.caption("Motor técnico experimental · datos públicos de Binance · actualización automática cada 30 s · no conecta cuentas ni ejecuta órdenes.")
st.markdown("**Datos:** precio en vivo aproximado por API pública; indicadores calculados con velas 1H/4H, algunas aún abiertas. No es una señal garantizada.")

symbol=st.sidebar.selectbox("Símbolo",["BTCUSDT"])
if st.sidebar.button("Actualizar ahora"):
    st.cache_data.clear()

@st.fragment(run_every="30s")
def live_dashboard():
  try:
    live_ticker=get_json(SPOT+"/api/v3/ticker/price", {"symbol":symbol})
    live_price=float(live_ticker["price"])
    d1,d4=current_data(symbol)
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
        st.info("WAIT — no hay una oportunidad que cumpla todos los filtros de la V1.2.")
    st.subheader("Precio BTC — 1H")
    chart=d1.set_index("open_time")[["close","ema55","ema200"]].tail(240)
    st.line_chart(chart)
    st.subheader("Backtest")
    st.write("Prueba inicial: 1H, entrada en la vela siguiente, stop 1 ATR, objetivo 2 ATR, comisión aproximada. No usar para decidir dinero real.")
    if st.button("Ejecutar backtest"):
        with st.spinner("Descargando histórico y calculando…"):
            end=int(datetime.now(timezone.utc).timestamp()*1000)
            start=int((datetime.now(timezone.utc)-timedelta(days=365*5)).timestamp()*1000)
            hist=paginate_klines(symbol,"1h",start,end,12000)
            res=[]
            for th in [6,7,8]:
                res.append((th,)+backtest(hist,th))
            bt=pd.DataFrame(res,columns=["Umbral","Trades","Wins","Losses","Win rate %","Resultado R","Max DD"])
            st.dataframe(bt.style.format({"Win rate %":"{:.1f}","Resultado R":"{:.1f}","Max DD":"{:.1%}"}),use_container_width=True)
  except Exception as e:
    st.error("No se pudieron cargar los datos. Comprueba la conexión o vuelve a actualizar.")
    st.caption(str(e))

live_dashboard()
