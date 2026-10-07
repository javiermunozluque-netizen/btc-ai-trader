# BUILD SYNC: 2026-10-07 V1.9.5

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="BTC AI Trader V1.9.5", page_icon="₿", layout="wide")

st.markdown("""<style>
:root{--btc-accent:#f7931a}
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1600px}
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(247,147,26,.075),transparent 34%),radial-gradient(ellipse at 95% 8%,rgba(85,119,255,.07),transparent 30%)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(127,140,160,.12),rgba(127,140,160,.035));border:1px solid rgba(127,140,160,.24);padding:17px 19px;border-radius:16px;min-height:108px}
[data-testid="stMetricLabel"]{font-size:.78rem;letter-spacing:.045em;text-transform:uppercase;opacity:.78}
[data-testid="stMetricValue"]{font-weight:750;letter-spacing:-.035em}
section[data-testid="stSidebar"]{border-right:1px solid rgba(127,140,160,.2)}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;font-weight:650;min-height:2.65rem}
div.stButton>button[kind="primary"]{background:var(--btc-accent);border-color:var(--btc-accent);color:#171717}
div[data-testid="stDataFrame"]{border:1px solid rgba(127,140,160,.22);border-radius:12px;overflow:hidden}
h1{font-weight:800;letter-spacing:-.055em} h2,h3{letter-spacing:-.03em}
@media(max-width:700px){.block-container{padding-top:.7rem;padding-left:1rem;padding-right:1rem}[data-testid="stMetric"]{padding:12px;min-height:94px}[data-testid="stMetricValue"]{font-size:1.35rem}}
</style>""",unsafe_allow_html=True)

SPOT="https://data-api.binance.vision"
FUT="https://fapi.binance.com"
TIMEOUT=15

@st.cache_data(ttl=20)
def get_json(url,params=None):
    r=requests.get(url,params=params,timeout=TIMEOUT); r.raise_for_status(); return r.json()

def klines(symbol="BTCUSDT",interval="1h",limit=1000,start=None,end=None):
    p={"symbol":symbol,"interval":interval,"limit":min(limit,1000)}
    if start is not None:p["startTime"]=int(start)
    if end is not None:p["endTime"]=int(end)
    data=get_json(SPOT+"/api/v3/klines",p)
    cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    df=pd.DataFrame(data,columns=cols)
    for c in cols[1:6]:df[c]=pd.to_numeric(df[c])
    df["open_time"]=pd.to_datetime(df.open_time,unit="ms",utc=True)
    return df[["open_time","open","high","low","close","volume","quote_volume","trades"]]

def paginate_klines(symbol,interval,start_ms,end_ms,max_rows=12000):
    out=[]; step={"1h":3600000,"4h":14400000}[interval]
    cur=max(int(start_ms),int(end_ms)-int(max_rows)*step)
    while cur<end_ms and sum(len(x) for x in out)<max_rows:
        batch=klines(symbol,interval,1000,cur,end_ms)
        if batch.empty:break
        out.append(batch)
        nxt=int(batch.open_time.iloc[-1].timestamp()*1000)+step
        if nxt<=cur:break
        cur=nxt
        if len(batch)<1000:break
    return pd.concat(out,ignore_index=True).drop_duplicates("open_time") if out else pd.DataFrame()

def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff();g=d.clip(lower=0);l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean();al=l.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan);return 100-(100/(1+rs))
def atr(df,n=14):
    pc=df.close.shift(1);tr=pd.concat([(df.high-df.low),(df.low-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def add_indicators(df):
    x=df.copy();x["ema55"]=ema(x.close,55);x["ema200"]=ema(x.close,200);x["rsi"]=rsi(x.close);x["atr"]=atr(x)
    x["vol_ma"]=x.volume.rolling(20).mean();x["vol_rel"]=x.volume/x.vol_ma
    ph=(x.high>x.high.shift(1))&(x.high>x.high.shift(2))&(x.high>=x.high.shift(-1))&(x.high>=x.high.shift(-2))
    pl=(x.low<x.low.shift(1))&(x.low<x.low.shift(2))&(x.low<=x.low.shift(-1))&(x.low<=x.low.shift(-2))
    x["ph"]=ph.shift(2).fillna(False).astype(bool);x["pl"]=pl.shift(2).fillna(False).astype(bool)
    x["pivot_high"]=x.high.shift(2).where(x.ph);x["pivot_low"]=x.low.shift(2).where(x.pl)
    return x

def structure_score(x):
    highs=x.loc[x.ph,"pivot_high"].tail(2).values;lows=x.loc[x.pl,"pivot_low"].tail(2).values
    hs="—";ls="—"
    if len(highs)==2:hs="HH" if highs[1]>highs[0] else "LH"
    if len(lows)==2:ls="HL" if lows[1]>lows[0] else "LL"
    if hs=="HH" and ls=="HL":return 2,hs,ls
    if hs=="LH" and ls=="LL":return -2,hs,ls
    return 0,hs,ls

@st.cache_data(show_spinner=False)
def prepare_diagnostic(df):
    x=add_indicators(df).reset_index(drop=True).copy()
    vals=np.zeros(len(x),dtype=int)
    for k in range(220,len(x)):vals[k]=structure_score(x.iloc[max(0,k-60):k+1])[0]
    x["structure_score"]=vals
    return x

def build_signals(df,start_index=220):
    x=prepare_diagnostic(df);n=len(x);s=max(220,int(start_index))
    close=x.close.to_numpy(float);op=x.open.to_numpy(float);ema55=x.ema55.to_numpy(float);ema200=x.ema200.to_numpy(float)
    rv=x.rsi.to_numpy(float);vr=x.vol_rel.to_numpy(float);atrv=x.atr.to_numpy(float);st=x.structure_score.to_numpy(int)
    trend=np.zeros(n,dtype=np.int8);conf=np.zeros(n,dtype=np.int8);mom=np.zeros(n,dtype=np.int8);vol=np.zeros(n,dtype=np.int8)
    valid=np.isfinite(atrv)&(atrv>0)&np.isfinite(ema200)&np.isfinite(rv)
    for i in range(s,n):
        trend[i]=2 if close[i]>ema55[i]>ema200[i] and ema55[i]>ema55[i-4] else (1 if close[i]>ema55[i] else (-2 if close[i]<ema55[i]<ema200[i] else -1))
        conf[i]=1 if close[i]>ema55[i] and ema55[i]>ema55[i-4] else (-1 if close[i]<ema55[i] and ema55[i]<ema55[i-4] else 0)
        mom[i]=1 if 50<=rv[i]<=65 and rv[i]>rv[i-3] else (-1 if rv[i]<45 else 0)
        vol[i]=1 if np.isfinite(vr[i]) and vr[i]>=1.2 and close[i]>op[i] else (-1 if np.isfinite(vr[i]) and vr[i]>=1.2 and close[i]<op[i] else 0)
    score=trend.astype(np.int16)+conf.astype(np.int16)+mom.astype(np.int16)+vol.astype(np.int16)+st.astype(np.int16)
    return x,{"trend":trend,"conf":conf,"mom":mom,"vol":vol,"structure":st,"score":score,"valid":valid}

def trade_outcomes(x,sig,fee_bps,slippage_bps,max_hold,start_index):
    n=len(x);score=sig["score"];valid=sig["valid"];op=x.open.to_numpy(float);high=x.high.to_numpy(float);low=x.low.to_numpy(float);close=x.close.to_numpy(float);atrv=x.atr.to_numpy(float)
    direction=np.where(score>0,1,np.where(score<0,-1,0)).astype(np.int8);cost_side=(fee_bps+slippage_bps)/10000.0;out={}
    for i in range(max(220,start_index),n-2):
        if direction[i]==0 or not valid[i]:continue
        d=int(direction[i]);entry_idx=i+1;entry=float(op[entry_idx]);risk=float(atrv[i]);stop=entry-d*risk;target=entry+d*2*risk
        last=min(entry_idx+int(max_hold),n-1);exit_idx=last;exit_price=float(close[last]);reason="Time exit"
        for j in range(entry_idx,last+1):
            if d==1:
                if low[j]<=stop:exit_idx=j;exit_price=stop;reason="Stop";break
                if high[j]>=target:exit_idx=j;exit_price=target;reason="Target";break
            else:
                if high[j]>=stop:exit_idx=j;exit_price=stop;reason="Stop";break
                if low[j]<=target:exit_idx=j;exit_price=target;reason="Target";break
        gross=d*(exit_price-entry)/risk;costs=2*cost_side*entry/risk;net=gross-costs
        fee=2*(fee_bps/10000.0*entry);slip=2*(slippage_bps/10000.0*entry)
        ambiguous=(d==1 and low[exit_idx]<=stop and high[exit_idx]>=target) or (d==-1 and high[exit_idx]>=stop and low[exit_idx]<=target)
        out[i]=(entry_idx,exit_idx,d,int(score[i]),entry,exit_price,stop,target,reason,gross,net,costs,fee,slip,fee+slip,bool(exit_idx==entry_idx),bool(ambiguous))
    return out

def evaluate_config(outcomes,sig,x,threshold,mode,dmode,fee_bps,risk_pct,start_index,return_trades=False):
    score=sig["score"];close=x.close.to_numpy(float);rv=x.rsi.to_numpy(float);vr=x.vol_rel.to_numpy(float);st=sig["structure"];open_time=x.open_time.to_numpy()
    eq=1.0;peak=1.0;maxdd=0.;wins=0;gp=0.;gl=0.;total=0.;last=-1;logs=[];count=0
    for i,data in outcomes.items():
        if i<start_index or i<=last or abs(int(score[i]))<threshold:continue
        ei,xi,d,sc,entry,exit_price,stop,target,reason,gross,net,costs,fee,slip,total_cost,same,amb=data
        if dmode=="LONG only" and d!=1:continue
        if dmode=="SHORT only" and d!=-1:continue
        if mode=="EMA trend" and not ((d==1 and close[i]>x.ema200.iloc[i]) or (d==-1 and close[i]<x.ema200.iloc[i])):continue
        if mode=="Momentum" and not ((d==1 and 50<=rv[i]<=65 and rv[i]>rv[i-3]) or (d==-1 and rv[i]<45)):continue
        if mode=="Volume" and not (np.isfinite(vr[i]) and vr[i]>=1.2):continue
        if mode=="Structure" and not ((d==1 and st[i]>0) or (d==-1 and st[i]<0)):continue
        count+=1;total+=net
        if net>0:wins+=1;gp+=net
        else:gl+=abs(net)
        eq*=max(0.,1+(risk_pct/100)*net);peak=max(peak,eq);maxdd=max(maxdd,(peak-eq)/peak if peak else 0);last=xi
        if return_trades:logs.append({"Entrada UTC":open_time[ei],"Salida UTC":open_time[xi],"Dirección":"LONG" if d==1 else "SHORT","Filtro":mode,"Score":sc,"Entrada":entry,"Salida":exit_price,"Stop inicial":stop,"Objetivo inicial":target,"Motivo salida":reason,"R bruto":gross,"Comisión (USDT)":fee,"Deslizamiento (USDT)":slip,"Costes (USDT)":total_cost,"Costes (R)":costs,"R neto":net,"Duración (h)":xi-ei,"Misma vela":same,"Vela ambigua":amb})
    res={"Trades":count,"Win rate %":wins/count*100 if count else 0.,"Profit factor":gp/gl if gl else (float("inf") if gp else 0.),"Expectancy R":total/count if count else 0.,"Net R":total,"Max DD %":maxdd*100,"Net return %":(eq-1)*100}
    return (res,pd.DataFrame(logs)) if return_trades else res

def true_filter_diagnostic(df,fee_bps,slippage_bps,risk_pct,max_hold,start_index=220):
    x,sig=build_signals(df,start_index);out=trade_outcomes(x,sig,fee_bps,slippage_bps,max_hold,start_index)
    thresholds=[3,4,5,6,7];modes=["Base","EMA trend","Momentum","Volume","Structure"];dirs=["Both","LONG only","SHORT only"]
    rows=[]
    for th in thresholds:
        for mode in modes:
            for dm in dirs:
                met=evaluate_config(out,sig,x,th,mode,dm,fee_bps,risk_pct,start_index)
                rows.append({"Umbral":th,"Filtro":mode,"Dirección":dm,**met})
    return pd.DataFrame(rows),x,sig,out

def render():
    st.title("₿ BTC AI Trader")
    st.caption("V1.9.5 · True Filter Diagnostic · Technical research only · No order execution")
    with st.sidebar:
        st.header("Parámetros")
        symbol=st.text_input("Símbolo","BTCUSDT")
        fee_bps=st.number_input("Comisión por lado (pb)",0.,50.,6.,.5)
        slippage_bps=st.number_input("Deslizamiento por lado (pb)",0.,50.,2.,.5)
        risk_pct=st.number_input("Riesgo por operación (%)",0.1,5.,0.5,.1)
        max_hold=st.number_input("Máximo de horas",1,168,48,1)
        st.caption("Los costes son solo para la simulación; no ejecuta órdenes.")

    st.markdown("### Motor de señal")
    d1,d4=current_data(symbol)
    s4,s1,mom,vol,stc,hs,ls,score=technical_score(d4,d1)
    c1,c2,c3,c4=st.columns(4);c1.metric("Score 1H",score);c2.metric("Tendencia 4H",s4);c3.metric("Momentum",mom);c4.metric("Estructura",stc)
    if score>=6:st.success("Sesgo LONG")
    elif score<=-6:st.error("Sesgo SHORT")
    else:st.info("WAIT · sin señal operativa")

    st.divider()
    st.markdown("### Backtest y validación")
    if st.button("Ejecutar True Filter Diagnostic V1.9.5",type="primary"):
        try:
            with st.spinner("Construyendo señales independientes y evaluando 75 combinaciones…"):
                end=int(datetime.now(timezone.utc).timestamp()*1000);start=int((datetime.now(timezone.utc)-timedelta(days=365*5)).timestamp()*1000)
                hist=paginate_klines(symbol,"1h",start,end,12000)
                if hist.empty or len(hist)<800:raise ValueError("No hay suficiente histórico.")
                cut=int(len(hist)*.70);dev=hist.iloc[:cut].copy();test=hist.copy()
                comp,_,_,_=true_filter_diagnostic(dev,fee_bps,slippage_bps,risk_pct,int(max_hold),0)
                ranked=comp.replace([np.inf,-np.inf],np.nan);eligible=ranked[ranked.Trades>=15]
                if eligible.empty:eligible=ranked[ranked.Trades>=5]
                if eligible.empty:eligible=ranked
                chosen=eligible.sort_values(["Expectancy R","Profit factor"],ascending=False).iloc[0]
                th=int(chosen.Umbral);mode=str(chosen.Filtro);dm=str(chosen.Dirección);label=f"Score {th} · {mode} · {dm}"
                oos_comp,x,sig,out=true_filter_diagnostic(test,fee_bps,slippage_bps,risk_pct,int(max_hold),cut)
                oos,trades=evaluate_config(out,sig,x,th,mode,dm,fee_bps,risk_pct,cut,True)
                st.session_state["tf_comp"]=comp;st.session_state["tf_oos"]=pd.DataFrame([{"Configuración elegida en desarrollo":label,**oos}]);st.session_state["tf_trades"]=trades
                st.session_state["tf_hist"]=len(hist);st.session_state["tf_cut"]=cut;st.session_state["tf_integrity"]=sig
                st.session_state.pop("tf_error",None)
        except Exception as e:
            st.session_state["tf_error"]=f"{type(e).__name__}: {e}"
    if st.session_state.get("tf_error"):st.error(st.session_state["tf_error"])
    if "tf_comp" in st.session_state:
        comp=st.session_state["tf_comp"];tr=st.session_state["tf_trades"]
        st.caption(f"Histórico {st.session_state['tf_hist']:,} velas · desarrollo {st.session_state['tf_cut']:,} · OOS {st.session_state['tf_hist']-st.session_state['tf_cut']:,}. 75 combinaciones: 5 umbrales × 5 filtros × 3 direcciones.")
        st.success("True Filter Diagnostic completado.")
        best=comp.replace([np.inf,-np.inf],np.nan).sort_values("Expectancy R",ascending=False).iloc[0]
        a,b,c,d=st.columns(4);a.metric("Mejor desarrollo",f"{best.Umbral} · {best.Filtro}");b.metric("Dirección",best.Dirección);c.metric("Expectativa",f"{best['Expectancy R']:.3f} R");d.metric("Trades",int(best.Trades))
        st.markdown("#### Comparación de filtros — desarrollo")
        view=comp.copy();view["Configuración"]=view.Filtro+" · "+view.Dirección
        st.dataframe(view.round({"Win rate %":1,"Profit factor":2,"Expectancy R":3,"Net R":2,"Max DD %":1,"Net return %":1}),use_container_width=True,hide_index=True)
        st.markdown("#### ¿Los filtros realmente filtran?")
        integ=comp.groupby(["Umbral","Dirección"]).agg(Configuraciones=("Filtro","nunique"),Min_trades=("Trades","min"),Max_trades=("Trades","max"),Min_expectativa=("Expectancy R","min"),Max_expectativa=("Expectancy R","max")).reset_index()
        st.dataframe(integ.round({"Min_expectativa":3,"Max_expectativa":3}),use_container_width=True,hide_index=True)
        st.info("En V1.9.5 los filtros son hipótesis independientes: Base usa solo score; EMA exige alineación con EMA200; Momentum exige RSI/momentum; Volume exige volumen relativo; Structure exige estructura a favor de la dirección. Por tanto, una diferencia de resultados ya no depende únicamente de alcanzar Score 7.")
        st.markdown("#### OOS — configuración elegida sin recalibración")
        st.dataframe(st.session_state["tf_oos"].round({"Win rate %":1,"Profit factor":2,"Expectancy R":3,"Net R":2,"Max DD %":1,"Net return %":1}),use_container_width=True,hide_index=True)
        if not tr.empty:
            trp=tr.sort_values("Salida UTC").copy();trp["R acumulado"]=trp["R neto"].cumsum()
            fig=go.Figure();fig.add_trace(go.Scatter(x=trp["Salida UTC"],y=trp["R acumulado"],mode="lines+markers",name="R acumulado"))
            fig.add_hline(y=0,line_dash="dash");fig.update_layout(height=320,margin=dict(l=8,r=8,t=15,b=8),xaxis_title="Salida UTC",yaxis_title="R acumulado",showlegend=False)
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
            st.dataframe(tr,use_container_width=True,hide_index=True)
        st.download_button("Descargar diagnóstico completo V1.9.5",comp.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_true_filter_diagnostic_v1_9_5.csv",mime="text/csv")
    st.divider()
    st.caption("V1.9.5 no pretende encontrar un resultado positivo a la fuerza: primero comprueba si los filtros aportan información incremental y después valida la configuración elegida en OOS.")

render()
