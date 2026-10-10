# BUILD SYNC: 2026-10-10 V1.16.1

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="BTC AI Trader V1.16.1", page_icon="₿", layout="wide")

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


def signal_close_horizon_analysis(df,start_index=220):
    x,sig=build_signals(df,start_index); score=sig["score"]; n=len(x); close=x.close.to_numpy(float)
    rows=[]
    for th in [3,4,5,6,7]:
        for dm in ["LONG","SHORT"]:
            mask=(np.abs(score)>=th) & (np.arange(n)>=max(220,start_index)); mask &= (score>0 if dm=="LONG" else score<0)
            idx=np.where(mask)[0]
            for h in [1,2,4,8,12,24,48]:
                v=idx[idx+h<n]; r=((close[v+h]/close[v])-1)*100; sr=r if dm=="LONG" else -r
                rows.append({"Umbral":th,"Dirección":dm,"Horizonte (h)":h,"Señales":len(v),"Retorno medio %":float(np.mean(sr)) if len(v) else 0.,"Mediana %":float(np.median(sr)) if len(v) else 0.,"Win rate %":float(np.mean(sr>0)*100) if len(v) else 0.})
    return pd.DataFrame(rows)

def regime_analysis(df,start_index=220):
    x,sig=build_signals(df,start_index); n=len(x); idx=np.arange(n); score=sig["score"]
    atr_pct=(x.atr/x.close*100).to_numpy(float); up=((x.close>x.ema200)&(x.ema55>x.ema200)).to_numpy(); down=((x.close<x.ema200)&(x.ema55<x.ema200)).to_numpy()
    vm=pd.Series(x.volume).rolling(48).median().to_numpy(); vh=x.volume.to_numpy(float)>vm
    regimes=np.full(n,"Rango",dtype=object); regimes[up]="Tendencia alcista"; regimes[down]="Tendencia bajista"
    base=max(220,start_index); hi=atr_pct>=np.nanpercentile(atr_pct[base:],70); lo=atr_pct<=np.nanpercentile(atr_pct[base:],30)
    rows=[]
    for reg,mask in [("Tendencia alcista",regimes=="Tendencia alcista"),("Tendencia bajista",regimes=="Tendencia bajista"),("Rango",regimes=="Rango"),("Alta volatilidad",hi),("Baja volatilidad",lo),("Volumen alto",vh),("Volumen normal/bajo",~vh)]:
        m=mask&(idx>=base); sc=score[m]
        rows.append({"Régimen":reg,"Velas":int(m.sum()),"Score medio":float(np.nanmean(sc)) if m.sum() else 0.,"Score +3 LONG":int((sc>=3).sum()),"Score -3 SHORT":int((sc<=-3).sum()),"Score +5 LONG":int((sc>=5).sum()),"Score -5 SHORT":int((sc<=-5).sum())})
    return pd.DataFrame(rows)

def execution_horizon_analysis(df,start_index=220):
    """
    Execution-aligned predictive diagnostic.
    Signal is generated on candle i close; entry is candle i+1 open.
    Horizon h means close of the h-th candle after entry.
    """
    x,sig=build_signals(df,start_index)
    score=sig["score"]; n=len(x); op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    rows=[]
    base=max(220,int(start_index))
    for th in [3,4,5,6,7]:
        for dm in ["LONG","SHORT"]:
            mask=(np.abs(score)>=th)&(np.arange(n)>=base)
            mask &= (score>0 if dm=="LONG" else score<0)
            idx=np.where(mask)[0]
            for h in [1,2,4,8,12,24,48]:
                entry_idx=idx+1
                exit_idx=entry_idx+h-1
                ok=(entry_idx<n)&(exit_idx<n)
                ei=entry_idx[ok]; xi=exit_idx[ok]
                r=((close[xi]/op[ei])-1)*100
                sr=r if dm=="LONG" else -r
                rows.append({
                    "Umbral":th,"Dirección":dm,"Horizonte (h)":h,"Señales":int(len(sr)),
                    "Retorno medio %":float(np.mean(sr)) if len(sr) else 0.,
                    "Mediana %":float(np.median(sr)) if len(sr) else 0.,
                    "Win rate %":float(np.mean(sr>0)*100) if len(sr) else 0.,
                    "P25 %":float(np.percentile(sr,25)) if len(sr) else 0.,
                    "P75 %":float(np.percentile(sr,75)) if len(sr) else 0.,
                    "Desv. estándar %":float(np.std(sr,ddof=1)) if len(sr)>1 else 0.,
                    "Mejor %":float(np.max(sr)) if len(sr) else 0.,
                    "Peor %":float(np.min(sr)) if len(sr) else 0.
                })
    return pd.DataFrame(rows)

def regime_horizon_analysis(df,start_index=220):
    """
    Execution-aligned returns by regime, threshold, direction and 24/48h horizon.
    Regime is measured on the signal candle; returns start at next open.
    """
    x,sig=build_signals(df,start_index)
    n=len(x); idx=np.arange(n); score=sig["score"]; op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    atr_pct=(x.atr/x.close*100).to_numpy(float)
    up=((x.close>x.ema200)&(x.ema55>x.ema200)).to_numpy()
    down=((x.close<x.ema200)&(x.ema55<x.ema200)).to_numpy()
    vm=pd.Series(x.volume).rolling(48).median().to_numpy()
    vh=x.volume.to_numpy(float)>vm
    regimes=np.full(n,"Rango",dtype=object)
    regimes[up]="Tendencia alcista"; regimes[down]="Tendencia bajista"
    base=max(220,int(start_index))
    hi=atr_pct>=np.nanpercentile(atr_pct[base:],70)
    lo=atr_pct<=np.nanpercentile(atr_pct[base:],30)
    regime_masks=[
        ("Tendencia alcista",regimes=="Tendencia alcista"),
        ("Tendencia bajista",regimes=="Tendencia bajista"),
        ("Rango",regimes=="Rango"),
        ("Alta volatilidad",hi),
        ("Baja volatilidad",lo),
        ("Volumen alto",vh),
        ("Volumen normal/bajo",~vh)
    ]
    rows=[]
    for reg,rmask in regime_masks:
        for th in [6,7]:
            for dm in ["LONG","SHORT"]:
                sigmask=rmask&(idx>=base)&(np.abs(score)>=th)
                sigmask &= (score>0 if dm=="LONG" else score<0)
                inds=np.where(sigmask)[0]
                for h in [24,48]:
                    ei=inds+1; xi=ei+h-1
                    ok=(ei<n)&(xi<n); ei=ei[ok]; xi=xi[ok]
                    r=((close[xi]/op[ei])-1)*100
                    sr=r if dm=="LONG" else -r
                    rows.append({
                        "Régimen":reg,"Umbral":th,"Dirección":dm,"Horizonte (h)":h,
                        "Señales":int(len(sr)),
                        "Retorno medio %":float(np.mean(sr)) if len(sr) else 0.,
                        "Mediana %":float(np.median(sr)) if len(sr) else 0.,
                        "Win rate %":float(np.mean(sr>0)*100) if len(sr) else 0.,
                        "P25 %":float(np.percentile(sr,25)) if len(sr) else 0.,
                        "P75 %":float(np.percentile(sr,75)) if len(sr) else 0.
                    })
    return pd.DataFrame(rows)

def edge_validation(df,start_index=220,boot_n=5000,seed=198):
    """
    Pre-specified statistical validation for the main hypotheses:
    Score 6/7 SHORT and LONG at 24/48h, using next-open execution.
    Bootstrap CI is a resampling diagnostic; random benchmark preserves
    direction and approximate signal frequency.
    """
    x,sig=build_signals(df,start_index)
    n=len(x); base=max(220,int(start_index)); score=sig["score"]
    op=x.open.to_numpy(float); close=x.close.to_numpy(float); rng=np.random.default_rng(seed)
    rows=[]; block_rows=[]; bench_rows=[]
    eligible=np.arange(base,n-49)
    for th in [6,7]:
        for dm in ["LONG","SHORT"]:
            sign=1 if dm=="LONG" else -1
            idx=np.where((np.abs(score)>=th)&(np.arange(n)>=base)&(np.arange(n)<n-49)&((score>0) if sign==1 else (score<0)))[0]
            for h in [24,48]:
                ei=idx+1; xi=ei+h-1
                ok=(ei<n)&(xi<n); ei=ei[ok]; xi=xi[ok]
                r=((close[xi]/op[ei])-1)*100*sign
                N=len(r)
                if N==0:
                    continue
                boot=rng.choice(r,size=(boot_n,N),replace=True).mean(axis=1)
                lo,hi=np.percentile(boot,[2.5,97.5])
                null=[]
                # Match the number of observations and direction, but ignore the score.
                for _ in range(2000):
                    ri=rng.choice(eligible,size=N,replace=False)
                    rei=ri+1; rxi=rei+h-1
                    rr=((close[rxi]/op[rei])-1)*100*sign
                    null.append(float(np.mean(rr)))
                null=np.asarray(null)
                null_mean=float(np.mean(null))
                p=float((1+np.sum(null>=float(np.mean(r))))/(len(null)+1))
                q95=float(np.percentile(null,95))
                rows.append({
                    "Umbral":th,"Dirección":dm,"Horizonte (h)":h,"Señales":N,
                    "Media %":float(np.mean(r)),"Mediana %":float(np.median(r)),
                    "Bootstrap IC95 inferior %":float(lo),"Bootstrap IC95 superior %":float(hi),
                    "Win rate %":float(np.mean(r>0)*100),
                    "Benchmark aleatorio media %":null_mean,
                    "Benchmark P95 %":q95,"p vs aleatorio":p,
                    "Ventaja vs benchmark %":float(np.mean(r)-null_mean)
                })
                # Four chronological blocks: stability, not another optimization.
                edges=np.linspace(0,N,5,dtype=int)
                for b in range(4):
                    rb=r[edges[b]:edges[b+1]]
                    block_rows.append({
                        "Umbral":th,"Dirección":dm,"Horizonte (h)":h,
                        "Bloque OOS":b+1,"Señales":len(rb),
                        "Media %":float(np.mean(rb)) if len(rb) else 0.,
                        "Mediana %":float(np.median(rb)) if len(rb) else 0.,
                        "Win rate %":float(np.mean(rb>0)*100) if len(rb) else 0.
                    })
    return pd.DataFrame(rows),pd.DataFrame(block_rows)


def robustness_analysis(df,start_index=220):
    """
    Pre-specified robustness checks for Score 6/7 SHORT at 24/48h:
    remove the top 1%, 5% and 10% observations and report the remaining mean.
    Also reports positive OOS blocks and worst block.
    """
    x,sig=build_signals(df,start_index)
    n=len(x); base=max(220,int(start_index)); score=sig["score"]
    op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    rows=[]; block_rows=[]
    for th in [6,7]:
        for h in [24,48]:
            idx=np.where((score<=-th)&(np.arange(n)>=base)&(np.arange(n)<n-49))[0]
            ei=idx+1; xi=ei+h-1; ok=(ei<n)&(xi<n); ei=ei[ok]; xi=xi[ok]
            r=((close[xi]/op[ei])-1)*-100
            if len(r)==0: continue
            s=np.sort(r)
            rows.append({
                "Umbral":th,"Dirección":"SHORT","Horizonte (h)":h,"Señales":len(r),
                "Media original %":float(np.mean(r)),
                "Sin top 1% %":float(np.mean(s[:max(1,int(np.floor(len(s)*.99)))])),
                "Sin top 5% %":float(np.mean(s[:max(1,int(np.floor(len(s)*.95)))])),
                "Sin top 10% %":float(np.mean(s[:max(1,int(np.floor(len(s)*.90)))])),
                "Mediana %":float(np.median(r)),
                "Peor %":float(np.min(r)),
                "Mejor %":float(np.max(r))
            })
            edges=np.linspace(0,len(r),5,dtype=int)
            for b in range(4):
                rb=r[edges[b]:edges[b+1]]
                block_rows.append({
                    "Umbral":th,"Dirección":"SHORT","Horizonte (h)":h,
                    "Bloque OOS":b+1,"Señales":len(rb),
                    "Media %":float(np.mean(rb)) if len(rb) else 0.,
                    "Mediana %":float(np.median(rb)) if len(rb) else 0.,
                    "Win rate %":float(np.mean(rb>0)*100) if len(rb) else 0.,
                    "Positivo":bool(np.mean(rb)>0) if len(rb) else False
                })
    return pd.DataFrame(rows),pd.DataFrame(block_rows)


def walk_forward_validation(df,initial_train_fraction=0.50,n_folds=5,min_train_signals=8,fee_bps=0.0,slippage_bps=0.0):
    """
    V1.15.0: gross-return expanding-window walk-forward; no fixed transaction costs.
    Compare predeclared configurations with/without a trend-alignment filter.
    Each signal's complete outcome must finish within its train/test segment.
    """
    x,sig=build_signals(df,220)
    n=len(x); score=np.asarray(sig["score"],dtype=float)
    op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    ema55=x.ema55.to_numpy(float); ema200=x.ema200.to_numpy(float)
    base=220
    if n-base < 500:
        raise ValueError("Histórico insuficiente para walk-forward (mínimo 500 velas tras indicadores).")
    if int(n_folds)<2: raise ValueError("El walk-forward necesita al menos dos folds.")
    initial_end=min(n-1,max(base+200,int(base+(n-base)*initial_train_fraction)))
    remaining=n-initial_end
    fold_size=max(1,remaining//int(n_folds))
    configs=[(th,dm,h,trend_filter) for th in (6,7) for dm in ("LONG","SHORT") for h in (24,48) for trend_filter in (False,True)]

    def get_returns(left,right,th,dm,h,trend_filter):
        # Signal at close i, entry at open i+1, exit at close i+h.
        lo=max(base,int(left)); hi=min(n-h,int(right)-h)
        if hi<=lo: return np.asarray([],dtype=float)
        idx=np.arange(lo,hi,dtype=int)
        sgn=1 if dm=="LONG" else -1
        mask=(np.abs(score[idx])>=th)&((score[idx]>0) if sgn==1 else (score[idx]<0))
        if trend_filter:
            trend_ok=(ema55[idx]>ema200[idx]) if sgn==1 else (ema55[idx]<ema200[idx])
            mask &= trend_ok
        raw=idx[mask]
        chosen=[]; next_allowed=-1
        for i in raw:
            if i>=next_allowed:
                chosen.append(int(i)); next_allowed=int(i)+h
        chosen=np.asarray(chosen,dtype=int)
        if not len(chosen): return np.asarray([],dtype=float)
        entry=chosen+1; exit_idx=entry+h-1
        valid=(entry<n)&(exit_idx<n)&(op[entry]>0)&(close[exit_idx]>0)
        return ((close[exit_idx[valid]] / op[entry[valid]]) - 1) * 100 * sgn

    rows=[]
    for fold in range(int(n_folds)):
        train_end=initial_end+fold*fold_size
        test_end=n if fold==int(n_folds)-1 else min(n,train_end+fold_size)
        if train_end>=n or test_end<=train_end: continue
        candidates=[]
        for th,dm,h,trend_filter in configs:
            train_r=get_returns(base,train_end,th,dm,h,trend_filter)
            if len(train_r)>=int(min_train_signals):
                candidates.append((float(np.mean(train_r)),len(train_r),th,dm,h,trend_filter,float(np.median(train_r))))
        if not candidates:
            rows.append({"Fold":fold+1,"Estado":"Sin configuración con muestra mínima",
                         "Fin entrenamiento":train_end,"Inicio test":train_end,"Fin test":test_end})
            continue
        # Choose using gross training mean only; test results do not affect selection.
        candidates.sort(key=lambda z:(z[0],z[1]),reverse=True)
        train_mean,train_n,th,dm,h,trend_filter,train_median=candidates[0]
        test_r=get_returns(train_end,test_end,th,dm,h,trend_filter)
        rows.append({
            "Fold":fold+1,"Estado":"OK" if len(test_r)>0 else "Sin señales OOS",
            "Umbral elegido":th,"Dirección elegida":dm,"Horizonte elegido (h)":h,
            "Filtro tendencia": "Alineado" if trend_filter else "Sin filtro",
            "Señales train":train_n,"Media train bruta %":train_mean,"Mediana train bruta %":train_median,
            "Señales test":len(test_r),
            "Media test OOS bruta %":float(np.mean(test_r)) if len(test_r) else np.nan,
            "Mediana test OOS bruta %":float(np.median(test_r)) if len(test_r) else np.nan,
            "Win rate test OOS bruta %":float(np.mean(test_r>0)*100) if len(test_r) else np.nan,
            "Suma retornos test OOS bruta %":float(np.sum(test_r)) if len(test_r) else np.nan,
            "Inicio entrenamiento":base,"Fin entrenamiento":train_end,
            "Inicio test":train_end,"Fin test":test_end,
            "Método":"Ventana expansiva; 16 variantes; selección solo train; sin costes"
        })
    result=pd.DataFrame(rows)
    if not result.empty and "Señales test" in result:
        valid=result[pd.to_numeric(result["Señales test"],errors="coerce").fillna(0)>0]
        summary={"Fold":"RESUMEN OOS","Estado":"Descriptivo; folds no independientes",
                 "Señales test":int(valid["Señales test"].sum()) if len(valid) else 0,
                 "Media test OOS bruta %":float(np.average(valid["Media test OOS bruta %"],weights=valid["Señales test"])) if len(valid) else np.nan,
                 "Suma retornos test OOS bruta %":float(valid["Suma retornos test OOS bruta %"].sum()) if len(valid) else np.nan,
                 "Método":"Ponderado por señales; retorno bruto; sin costes; no compuesto"}
        result=pd.concat([result,pd.DataFrame([summary])],ignore_index=True)
    return result


def variant_diagnostic_validation(df,initial_train_fraction=0.50,n_folds=5,random_reps=500,boot_reps=1000,seed=1515):
    """
    V1.15.0: fixed OOS blocks, random-time benchmark with matching direction/horizon/count,
    block-bootstrap confidence intervals and Holm adjustment across the 16 variants.
    All returns are gross; no transaction costs are deducted.
    """
    x,sig=build_signals(df,220)
    n=len(x); score=np.asarray(sig["score"],dtype=float)
    op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    ema55=x.ema55.to_numpy(float); ema200=x.ema200.to_numpy(float)
    base=220
    if n-base<500: raise ValueError("Histórico insuficiente para diagnóstico (mínimo 500 velas tras indicadores).")
    if int(n_folds)<2: raise ValueError("Se necesitan al menos dos bloques temporales.")
    initial_end=min(n-1,max(base+200,int(base+(n-base)*initial_train_fraction)))
    remaining=n-initial_end; fold_size=max(1,remaining//int(n_folds))
    configs=[(th,dm,h,tf) for th in (6,7) for dm in ("LONG","SHORT") for h in (24,48) for tf in (False,True)]
    rng=np.random.default_rng(seed)
    rows=[]
    for fold in range(int(n_folds)):
        left=initial_end+fold*fold_size
        right=n if fold==int(n_folds)-1 else min(n,left+fold_size)
        if left>=n or right<=left: continue
        bh=np.nan
        if right-left>=2 and op[left]>0 and close[right-1]>0:
            bh=(close[right-1]/op[left]-1)*100
        for th,dm,h,tf in configs:
            lo=max(base,left); hi=min(n-h,right-h)
            idxs=np.arange(lo,hi,dtype=int) if hi>lo else np.asarray([],dtype=int)
            sgn=1 if dm=="LONG" else -1
            mask=(np.abs(score[idxs])>=th)&((score[idxs]>0) if sgn==1 else (score[idxs]<0))
            if tf and len(idxs):
                mask &= (ema55[idxs]>ema200[idxs]) if sgn==1 else (ema55[idxs]<ema200[idxs])
            raw=idxs[mask]; chosen=[]; next_allowed=-1
            for j in raw:
                if j>=next_allowed:
                    chosen.append(int(j)); next_allowed=int(j)+h
            chosen=np.asarray(chosen,dtype=int)
            entry=chosen+1; exit_idx=entry+h-1
            valid=(entry<n)&(exit_idx<right)&(op[entry]>0)&(close[exit_idx]>0)
            chosen=chosen[valid]; entry=entry[valid]; exit_idx=exit_idx[valid]
            rets=((close[exit_idx]/op[entry])-1)*100*sgn if len(chosen) else np.asarray([],dtype=float)
            pval=np.nan; rand_mean=np.nan; valid_reps=0
            if len(chosen):
                pool=np.arange(max(base,left),min(n-h,right-h),dtype=int)
                null=[]
                for _ in range(int(random_reps)):
                    order=rng.permutation(pool); picks=[]
                    for cand in order:
                        ci=int(cand)
                        if all(abs(ci-old)>=h for old in picks):
                            picks.append(ci)
                            if len(picks)>=len(chosen): break
                    if len(picks)!=len(chosen): continue
                    picks=np.asarray(sorted(picks),dtype=int)
                    ent=picks+1; ex=ent+h-1
                    if np.any(ex>=right) or np.any(ent>=n) or np.any(op[ent]<=0) or np.any(close[ex]<=0): continue
                    rr=((close[ex]/op[ent])-1)*100*sgn
                    null.append(float(np.mean(rr)))
                valid_reps=len(null)
                if valid_reps:
                    rand_mean=float(np.mean(null))
                    pval=float((1+sum(v>=float(np.mean(rets)) for v in null))/(valid_reps+1))
            # Moving-block bootstrap of the chronological outcome sequence (exploratory CI).
            ci_low=np.nan; ci_high=np.nan
            if len(rets)>=3:
                L=min(3,len(rets)); blocks=int(np.ceil(len(rets)/L))
                starts=rng.integers(0,len(rets),size=(int(boot_reps),blocks))
                sampled=np.concatenate([np.take(rets,(starts[:,j,None]+np.arange(L))%len(rets),axis=0) for j in range(blocks)],axis=1)[:,:len(rets)]
                means=np.mean(sampled,axis=1)
                ci_low=float(np.percentile(means,2.5)); ci_high=float(np.percentile(means,97.5))
            rows.append({
                "Fold":fold+1,"Umbral":th,"Dirección":dm,"Horizonte (h)":h,
                "Filtro tendencia":"Alineado" if tf else "Sin filtro",
                "Señales OOS":len(rets),
                "Media OOS bruta %":float(np.mean(rets)) if len(rets) else np.nan,
                "Mediana OOS bruta %":float(np.median(rets)) if len(rets) else np.nan,
                "Win rate OOS %":float(np.mean(rets>0)*100) if len(rets) else np.nan,
                "Suma retornos OOS %":float(np.sum(rets)) if len(rets) else np.nan,
                "IC95 bootstrap low %":ci_low,"IC95 bootstrap high %":ci_high,
                "Buy & hold fold %":bh,"Media benchmark aleatorio %":rand_mean,
                "Ventaja vs aleatorio pp":float(np.mean(rets)-rand_mean) if len(rets) and np.isfinite(rand_mean) else np.nan,
                "p unilateral exploratorio":pval,"Réplicas aleatorias válidas":valid_reps,
                "Costes descontados":False,
                "Nota":"IC bootstrap aproximado; p no ajustado en fila por bloque"
            })
    result=pd.DataFrame(rows)
    if result.empty: return result
    # Aggregate variant results across OOS blocks using all trades, with no test-based tuning.
    actual=result[result["Fold"].apply(lambda z:isinstance(z,(int,np.integer)))].copy()
    summaries=[]
    for (th,dm,h,tf),g in actual.groupby(["Umbral","Dirección","Horizonte (h)","Filtro tendencia"],dropna=False):
        gg=g[g["Señales OOS"]>0]
        total=int(gg["Señales OOS"].sum()) if len(gg) else 0
        summaries.append({
            "Fold":"RESUMEN variante","Umbral":th,"Dirección":dm,"Horizonte (h)":h,"Filtro tendencia":tf,
            "Señales OOS":total,
            "Media OOS bruta %":float(np.average(gg["Media OOS bruta %"],weights=gg["Señales OOS"])) if total else np.nan,
            "Mediana OOS bruta %":np.nan,
            "Win rate OOS %":np.nan,
            "Suma retornos OOS %":float(gg["Suma retornos OOS %"].sum()) if total else np.nan,
            "IC95 bootstrap low %":np.nan,"IC95 bootstrap high %":np.nan,
            "Buy & hold fold %":np.nan,
            "Media benchmark aleatorio %":float(np.average(gg["Media benchmark aleatorio %"].dropna(),weights=gg.loc[gg["Media benchmark aleatorio %"].notna(),"Señales OOS"])) if gg["Media benchmark aleatorio %"].notna().any() else np.nan,
            "Ventaja vs aleatorio pp":float(np.average(gg["Ventaja vs aleatorio pp"].dropna(),weights=gg.loc[gg["Ventaja vs aleatorio pp"].notna(),"Señales OOS"])) if gg["Ventaja vs aleatorio pp"].notna().any() else np.nan,
            "p unilateral exploratorio":float(max(gg["p unilateral exploratorio"].dropna())) if gg["p unilateral exploratorio"].notna().any() else np.nan,
            "Réplicas aleatorias válidas":int(gg["Réplicas aleatorias válidas"].sum()),
            "Costes descontados":False,
            "Nota":"Resumen agregado OOS; p global conservador = máximo p por bloque disponible"
        })
    summary=pd.DataFrame(summaries)
    # Holm step-down correction across 16 variant-level summary p-values.
    pcol="p unilateral exploratorio"
    valid_idx=summary.index[summary[pcol].notna()].tolist()
    ordered=sorted(valid_idx,key=lambda i:float(summary.loc[i,pcol]))
    m=len(ordered); running=0.0
    summary["p Holm (16 variantes)"]=np.nan
    for rank,i in enumerate(ordered):
        adj=min(1.0,(m-rank)*float(summary.loc[i,pcol]))
        running=max(running,adj)
        summary.loc[i,"p Holm (16 variantes)"]=running
    result["p Holm (16 variantes)"]=np.nan
    result=pd.concat([result,summary.reindex(columns=result.columns)],ignore_index=True,sort=False)
    return result



def confirmatory_holdout_test(df, holdout_fraction=0.20, random_reps=2000, boot_reps=3000, seed=1616):
    """V1.16.1: one predeclared Score >=7 LONG, 48h test on final chronological holdout."""
    x, sig = build_signals(df, 220)
    n=len(x); base=220; h=48
    if n-base < 1000: raise ValueError("Se necesitan al menos 1.000 velas tras indicadores para la prueba confirmatoria.")
    split=max(base+500,int(n*(1.0-float(holdout_fraction))))
    if split>=n-100: raise ValueError("No queda suficiente periodo reservado para el test.")
    score=np.asarray(sig["score"],dtype=float); op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    times=pd.to_datetime(x.open_time,utc=True); month=times.dt.strftime("%Y-%m").to_numpy()
    candidates=np.arange(split,n-h,dtype=int)
    candidates=candidates[(score[candidates]>=7)&(op[candidates+1]>0)&(close[candidates+h]>0)]
    selected=[]; next_allowed=split
    for i in candidates:
        if i>=next_allowed:
            selected.append(int(i)); next_allowed=int(i)+h
    selected=np.asarray(selected,dtype=int)
    if len(selected)<5: raise ValueError(f"Solo hay {len(selected)} operaciones válidas en el holdout; se necesitan al menos 5.")
    def returns_at(ix):
        ix=np.asarray(ix,dtype=int)
        return (close[ix+h]/op[ix+1]-1.0)*100.0
    observed=returns_at(selected)
    observed_months={}
    for i in selected: observed_months[month[i]]=observed_months.get(month[i],0)+1
    pools={}
    for i in range(split,n-h):
        if op[i+1]>0 and close[i+h]>0: pools.setdefault(month[i],[]).append(i)
    pool_arrays={k:np.asarray(v,dtype=int) for k,v in pools.items()}
    rng=np.random.default_rng(seed); null_means=[]; target=max(100,int(random_reps)*0.10)
    attempts=0; max_attempts=int(random_reps)*12
    while len(null_means)<int(random_reps) and attempts<max_attempts:
        attempts+=1; parts=[]; feasible=True
        for m,need in observed_months.items():
            pool=pool_arrays.get(m,np.asarray([],dtype=int))
            if len(pool)<need: feasible=False; break
            parts.append(rng.choice(pool,size=need,replace=False))
        if not feasible: continue
        chosen=np.sort(np.concatenate(parts))
        if len(chosen)>1 and np.any(np.diff(chosen)<h): continue
        null_means.append(float(np.mean(returns_at(chosen))))
    if len(null_means)<target:
        raise ValueError(f"Benchmark comparable insuficiente: solo {len(null_means)} réplicas válidas.")
    null=np.asarray(null_means); mean_obs=float(np.mean(observed))
    p_value=float((1+np.sum(null>=mean_obs))/(len(null)+1))
    block=min(3,len(observed)); boot_means=np.empty(int(boot_reps),dtype=float)
    for b in range(int(boot_reps)):
        sampled=[]
        while len(sampled)<len(observed):
            start=int(rng.integers(0,len(observed)))
            sampled.extend(observed[(start+np.arange(block))%len(observed)].tolist())
        boot_means[b]=float(np.mean(sampled[:len(observed)]))
    ci_lo,ci_hi=np.percentile(boot_means,[2.5,97.5]); null_lo,null_hi=np.percentile(null,[2.5,97.5])
    out=pd.DataFrame([{
        "Hipótesis predefinida":"Score >= 7 LONG · 48h",
        "Inicio holdout":str(times.iloc[split].date()),"Fin holdout":str(times.iloc[-1].date()),
        "Operaciones OOS":int(len(observed)),"Media bruta %":mean_obs,
        "Mediana bruta %":float(np.median(observed)),"Win rate %":float(np.mean(observed>0)*100),
        "IC95 MBB inferior %":float(ci_lo),"IC95 MBB superior %":float(ci_hi),
        "Benchmark aleatorio media %":float(np.mean(null)),"Benchmark aleatorio P2.5 %":float(null_lo),
        "Benchmark aleatorio P97.5 %":float(null_hi),"Ventaja vs benchmark pp":float(mean_obs-np.mean(null)),
        "p unilateral vs aleatorio":p_value,"Réplicas aleatorias válidas":int(len(null)),
        "Costes descontados":"No · retorno bruto",
        "Interpretación":"Evidencia compatible con edge" if p_value<0.05 and ci_lo>0 else "Sin evidencia confirmatoria suficiente"
    }])
    trades=pd.DataFrame({"Entrada UTC":times.iloc[selected+1].to_numpy(),"Salida UTC":times.iloc[selected+h].to_numpy(),
        "Dirección":"LONG","Score señal":score[selected].astype(int),"Precio entrada":op[selected+1],
        "Precio salida":close[selected+h],"Retorno bruto %":observed,"Horizonte (h)":h})
    return out,trades



def temporal_validation(df,start_index=220,boot_n=4000,seed=1102,block_len=3):
    """
    V1.15.0: audited temporal validation with genuinely randomized benchmark sets.
    Outcome windows are non-overlapping. Benchmark samples timestamps randomly
    within the same chronological quartiles, matching each quartile's signal count
    and enforcing non-overlap globally. Failed/infeasible replications are counted.
    Four chronological segments are descriptive stability checks, not retrained
    walk-forward tests. Benchmark p-values are exploratory, not proof of alpha.
    """
    x,sig=build_signals(df,start_index)
    n=len(x); base=max(220,int(start_index)); score=np.asarray(sig["score"],dtype=float)
    op=x.open.to_numpy(float); close=x.close.to_numpy(float)
    rng=np.random.default_rng(seed)
    rows=[]; walk=[]

    def non_overlapping(candidates,h):
        chosen=[]; next_allowed=-1
        for ix in np.sort(np.asarray(candidates,dtype=int)):
            if ix>=next_allowed:
                chosen.append(int(ix)); next_allowed=int(ix)+h
        return np.asarray(chosen,dtype=int)

    def returns_at(indices,h,sign):
        if len(indices)==0: return np.asarray([],dtype=float)
        ei=indices+1; xi=ei+h-1
        valid=(ei<n)&(xi<n)&(op[ei]>0)&(close[xi]>0)
        return ((close[xi[valid]]/op[ei[valid]])-1)*100*sign

    def random_matched_set(pools,counts,h,max_tries=30):
        """Randomized backtracking-by-restarts; enforce spacing across quartile edges."""
        for _ in range(max_tries):
            chosen=[]
            order=rng.permutation(4)
            ok=True
            for q in order:
                cnt=counts[int(q)]
                if cnt==0: continue
                candidates=rng.permutation(pools[int(q)])
                picked=[]
                for cand in candidates:
                    cand=int(cand)
                    if all(abs(cand-prev)>=h for prev in chosen) and all(abs(cand-prev)>=h for prev in picked):
                        picked.append(cand)
                        if len(picked)==cnt: break
                if len(picked)<cnt:
                    ok=False; break
                chosen.extend(picked)
            if ok and len(chosen)==sum(counts) and len(chosen)>0:
                chosen=np.asarray(sorted(chosen),dtype=int)
                if len(chosen)<2 or np.all(np.diff(chosen)>=h):
                    return chosen
        return None

    for th in [6,7]:
        for dm in ["LONG","SHORT"]:
            sign=1 if dm=="LONG" else -1
            for h in [24,48]:
                raw=np.where((np.abs(score)>=th)&(np.arange(n)>=base)&
                             (np.arange(n)<n-h-1)&
                             ((score>0) if sign==1 else (score<0)))[0]
                inds=non_overlapping(raw,h)
                r=returns_at(inds,h,sign); N=len(r)
                if N<8: continue

                L=max(1,min(int(block_len),max(1,N//2)))
                nblocks=int(np.ceil(N/L))
                starts=rng.integers(0,N,size=(boot_n,nblocks))
                samples=np.concatenate(
                    [np.take(r,(starts[:,j,None]+np.arange(L))%N)
                     for j in range(nblocks)],axis=1)[:,:N]
                boot_means=samples.mean(axis=1)
                lo,hi=np.percentile(boot_means,[2.5,97.5])

                # Pools are all valid timestamps in each quartile, independent of score.
                pools=[]; counts=[]
                quartile_ids=np.minimum(3,((inds-base)/max(1,n-base)*4).astype(int))
                for q in range(4):
                    left=base+int((n-base)*q/4)
                    right=min(n-h-1,base+int((n-base)*(q+1)/4))
                    pools.append(np.arange(max(base,left),max(max(base,left),right),dtype=int))
                    counts.append(int(np.sum(quartile_ids==q)))

                nullmeans=[]; attempts=2000
                for _ in range(attempts):
                    picks=random_matched_set(pools,counts,h)
                    if picks is None: continue
                    rr=returns_at(picks,h,sign)
                    if len(rr)==N: nullmeans.append(float(np.mean(rr)))
                null=np.asarray(nullmeans,dtype=float)
                observed=float(np.mean(r))
                null_mean=float(np.mean(null)) if len(null) else np.nan
                null_median=float(np.median(null)) if len(null) else np.nan
                p=float((1+np.sum(null>=observed))/(len(null)+1)) if len(null) else np.nan

                rows.append({
                    "Umbral":th,"Dirección":dm,"Horizonte (h)":h,
                    "Señales":N,"Señales brutas":len(raw),
                    "Media %":observed,"Mediana %":float(np.median(r)),
                    "MBB IC95 inferior %":float(lo),"MBB IC95 superior %":float(hi),
                    "Win rate %":float(np.mean(r>0)*100),
                    "Benchmark temporal media %":null_mean,
                    "Benchmark temporal mediana %":null_median,
                    "Ventaja vs benchmark %":float(observed-null_mean) if len(null) else np.nan,
                    "p vs benchmark temporal":p,
                    "Réplicas benchmark válidas":len(null),
                    "Réplicas benchmark fallidas":attempts-len(null),
                    "Longitud bloque observaciones":L,
                    "Ventanas solapadas excluidas":int(len(raw)-N),
                    "Separación mínima temporal (h)":h
                })

                edges=np.linspace(0,N,5,dtype=int)
                for k in range(4):
                    z=r[edges[k]:edges[k+1]]
                    walk.append({
                        "Umbral":th,"Dirección":dm,"Horizonte (h)":h,
                        "Bloque temporal":k+1,"Señales":len(z),
                        "Media %":float(np.mean(z)) if len(z) else np.nan,
                        "Mediana %":float(np.median(z)) if len(z) else np.nan,
                        "Win rate %":float(np.mean(z>0)*100) if len(z) else np.nan,
                        "Media positiva":bool(np.mean(z)>0) if len(z) else False,
                        "Interpretación":"Bloque cronológico descriptivo; sin reentrenamiento"
                    })
    return pd.DataFrame(rows),pd.DataFrame(walk)

def technical_score(df4,df1):
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

def render():
    st.title("₿ BTC AI Trader")
    st.caption("V1.16.1 · Confirmatory Holdout Research · Technical research only · No order execution")
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
    if st.button("Ejecutar validación confirmatoria V1.16.1",type="primary"):
        try:
            with st.spinner("Validando entrada next-open, horizontes, distribución, regímenes y 75 combinaciones…"):
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
                st.session_state["tf_horizon_close_dev"]=signal_close_horizon_analysis(dev,0)
                st.session_state["tf_horizon_close_oos"]=signal_close_horizon_analysis(test,cut)
                st.session_state["tf_horizon_exec_dev"]=execution_horizon_analysis(dev,0)
                st.session_state["tf_horizon_exec_oos"]=execution_horizon_analysis(test,cut)
                st.session_state["tf_regime_dev"]=regime_analysis(dev,0)
                st.session_state["tf_regime_oos"]=regime_analysis(test,cut)
                st.session_state["tf_regime_horizon_dev"]=regime_horizon_analysis(dev,0)
                st.session_state["tf_regime_horizon_oos"]=regime_horizon_analysis(test,cut)
                st.session_state["tf_edge_oos"],st.session_state["tf_edge_blocks_oos"]=edge_validation(test,cut,5000,198)
                st.session_state["tf_robust_oos"],st.session_state["tf_robust_blocks_oos"]=robustness_analysis(test,cut)
                st.session_state["tf_temporal_oos"],st.session_state["tf_walk_oos"]=temporal_validation(test,cut,4000,seed=1100,block_len=3)
                st.session_state["tf_walk_forward"]=walk_forward_validation(hist,initial_train_fraction=0.50,n_folds=5,min_train_signals=8)
                st.session_state["tf_variant_diagnostic"]=variant_diagnostic_validation(hist,initial_train_fraction=0.50,n_folds=5,random_reps=500,boot_reps=1000,seed=1515)
                try:
                    st.session_state["tf_confirmatory"],st.session_state["tf_confirmatory_trades"]=confirmatory_holdout_test(hist,holdout_fraction=0.20,random_reps=2000,boot_reps=3000,seed=1616)
                    st.session_state.pop("tf_confirmatory_error",None)
                except Exception as ce:
                    st.session_state["tf_confirmatory"]=None
                    st.session_state["tf_confirmatory_trades"]=pd.DataFrame()
                    st.session_state["tf_confirmatory_error"]=f"{type(ce).__name__}: {ce}"
                st.session_state.pop("tf_error",None)
        except Exception as e:
            st.session_state["tf_error"]=f"{type(e).__name__}: {e}"
    if st.session_state.get("tf_error"):st.error(st.session_state["tf_error"])
    if "tf_comp" in st.session_state:
        comp=st.session_state["tf_comp"];tr=st.session_state["tf_trades"]
        st.caption(f"Histórico {st.session_state['tf_hist']:,} velas · desarrollo {st.session_state['tf_cut']:,} · OOS {st.session_state['tf_hist']-st.session_state['tf_cut']:,}. 75 combinaciones + validación next-open + distribución + régimen + prueba confirmatoria reservada.")
        st.success("True Filter Diagnostic completado.")
        best=comp.replace([np.inf,-np.inf],np.nan).sort_values("Expectancy R",ascending=False).iloc[0]
        a,b,c,d=st.columns(4);a.metric("Mejor desarrollo",f"{best.Umbral} · {best.Filtro}");b.metric("Dirección",best.Dirección);c.metric("Expectativa",f"{best['Expectancy R']:.3f} R");d.metric("Trades",int(best.Trades))
        st.markdown("#### Comparación de filtros — desarrollo")
        view=comp.copy();view["Configuración"]=view.Filtro+" · "+view.Dirección
        st.dataframe(view.round({"Win rate %":1,"Profit factor":2,"Expectancy R":3,"Net R":2,"Max DD %":1,"Net return %":1}),use_container_width=True,hide_index=True)
        st.markdown("#### 🔬 Predictive Edge — validación con entrada realista")
        st.caption("V1.15.0 alinea el diagnóstico con la ejecución: señal al cierre de la vela i → entrada en apertura de i+1. El horizonte h mide el cierre de la h.ª vela desde esa entrada.")
        hv=st.session_state["tf_horizon_exec_oos"]
        piv=hv.pivot_table(index=["Umbral","Dirección"],columns="Horizonte (h)",values="Retorno medio %",aggfunc="first").reset_index()
        st.dataframe(piv.round(3),use_container_width=True,hide_index=True)
        besth=hv.sort_values(["Retorno medio %","Señales"],ascending=[False,False]).iloc[0]
        bu=int(besth["Umbral"]); bd=str(besth["Dirección"]); bh=int(besth["Horizonte (h)"])
        br=float(besth["Retorno medio %"]); bn=int(besth["Señales"])
        st.success(f"Mejor retorno medio OOS: Score {bu} {bd} a {bh}h → {br:.3f}% por señal (N={bn}).")

        st.markdown("#### 📊 Robustez de la distribución OOS")
        dist=hv[(hv["Umbral"].isin([6,7]))&(hv["Dirección"].isin(["LONG","SHORT"]))&(hv["Horizonte (h)"].isin([24,48]))].copy()
        st.dataframe(dist[["Umbral","Dirección","Horizonte (h)","Señales","Retorno medio %","Mediana %","Win rate %","P25 %","P75 %","Desv. estándar %","Mejor %","Peor %"]].round(3),use_container_width=True,hide_index=True)
        st.caption("La media puede ocultar colas. Mediana, P25/P75, win rate y peor resultado ayudan a comprobar si el efecto está repartido entre muchas señales o concentrado en unas pocas.")

        st.markdown("#### 🔎 Comparación: cierre de señal vs entrada next-open")
        hc=st.session_state["tf_horizon_close_oos"]
        hc24=hc[(hc["Umbral"].isin([6,7]))&(hc["Dirección"].isin(["LONG","SHORT"]))&(hc["Horizonte (h)"].isin([24,48]))][["Umbral","Dirección","Horizonte (h)","Señales","Retorno medio %"]].copy()
        he24=hv[(hv["Umbral"].isin([6,7]))&(hv["Dirección"].isin(["LONG","SHORT"]))&(hv["Horizonte (h)"].isin([24,48]))][["Umbral","Dirección","Horizonte (h)","Señales","Retorno medio %"]].copy()
        hc24["Medición"]="Cierre señal → cierre futuro"; he24["Medición"]="Next-open → cierre futuro"
        st.dataframe(pd.concat([hc24,he24],ignore_index=True).round(3),use_container_width=True,hide_index=True)

        st.markdown("#### 🌐 Regímenes de mercado — OOS")
        st.dataframe(st.session_state["tf_regime_oos"],use_container_width=True,hide_index=True)
        st.markdown("#### 🧩 Señal + régimen + horizonte — OOS")
        rh=st.session_state["tf_regime_horizon_oos"]
        rhv=rh[(rh["Umbral"].isin([6,7]))&(rh["Horizonte (h)"].isin([24,48]))].copy()
        st.dataframe(rhv.round(3),use_container_width=True,hide_index=True)
        st.info("V1.15.0 no cambia TP/SL ni fuerza una estrategia positiva: primero valida si el efecto predictivo sobrevive al next-open, a la distribución, al horizonte y al régimen.")
        
        st.markdown("#### 🧭 V1.15.0 — Walk-forward real con ventana expansiva")
        st.caption("En cada fold se elige entre 16 variantes usando solo el entrenamiento y retornos brutos; se evalúa en el siguiente bloque. No se descuentan costes fijos ni variables. Incluye filtro opcional de alineación EMA55/EMA200. Resumen descriptivo.")
        st.dataframe(st.session_state["tf_walk_forward"].round(3),use_container_width=True,hide_index=True)

        st.markdown("#### 🎯 V1.16.1 — Prueba confirmatoria en periodo reservado")
        st.caption("Una sola hipótesis predefinida: Score >= 7 LONG, horizonte 48h. Se reserva cronológicamente el último 20% del histórico. Benchmark aleatorio con igual número de operaciones por mes, mismo horizonte y operaciones no solapadas. Retornos brutos, sin costes.")
        conf=st.session_state.get("tf_confirmatory")
        if conf is not None and not conf.empty:
            cr=conf.iloc[0]
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Operaciones reservadas",int(cr["Operaciones OOS"]))
            m2.metric("Media bruta",f'{cr["Media bruta %"]:.3f}%')
            m3.metric("Ventaja vs aleatorio",f'{cr["Ventaja vs benchmark pp"]:+.3f} pp')
            m4.metric("p unilateral",f'{cr["p unilateral vs aleatorio"]:.4f}')
            st.dataframe(conf.round(4),use_container_width=True,hide_index=True)
            if cr["p unilateral vs aleatorio"]<0.05 and cr["IC95 MBB inferior %"]>0:
                st.success("La hipótesis supera los dos criterios predefinidos. Aun así, debe replicarse en datos futuros.")
            else:
                st.warning("La hipótesis NO queda confirmada con estos datos. No la conviertas en estrategia operativa ni pruebes variantes nuevas sobre este mismo holdout.")
            st.markdown("##### Operaciones del periodo reservado")
            st.dataframe(st.session_state["tf_confirmatory_trades"].round(4),use_container_width=True,hide_index=True)
            st.download_button("Descargar prueba confirmatoria V1.16.1",conf.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_confirmatory_v1_16_1.csv",mime="text/csv")
        elif st.session_state.get("tf_confirmatory_error"):
            st.warning("El análisis general se ha completado, pero el test confirmatorio no pudo calcularse: " + st.session_state["tf_confirmatory_error"])
        else:
            st.info("Pulsa ejecutar para calcular el test confirmatorio.")
        
        st.markdown("#### 🔬 V1.15.0 — Diagnóstico estadístico de las 16 variantes")
        st.caption("Todas las variantes se evalúan en los mismos bloques OOS. Retornos brutos sin costes. Se añaden IC bootstrap aproximados y p ajustados por Holm en el resumen; las pruebas siguen siendo exploratorias y sensibles a la dependencia temporal.")
        st.dataframe(st.session_state["tf_variant_diagnostic"].round(3),use_container_width=True,hide_index=True)

        st.markdown("#### 🧪 V1.15.0 — Validación estadística del edge")
        st.caption("Hipótesis predefinidas: Score 6/7, LONG/SHORT, horizontes 24/48h. El benchmark aleatorio mantiene la dirección y el número de señales, pero elimina el criterio de score.")
        edge=st.session_state["tf_edge_oos"].copy()
        st.dataframe(edge.round(3),use_container_width=True,hide_index=True)
        st.caption("IC95% por bootstrap de las señales observadas. El p vs aleatorio es un test de referencia frente a entradas aleatorias con la misma dirección y frecuencia; no sustituye un test estadístico completo y las señales solapadas pueden introducir dependencia.")
        st.markdown("##### 📈 Estabilidad temporal OOS")
        eb=st.session_state["tf_edge_blocks_oos"].copy()
        st.dataframe(eb.round(3),use_container_width=True,hide_index=True)
        focus=edge[(edge["Dirección"]=="SHORT")&(edge["Horizonte (h)"].isin([24,48]))&(edge["Umbral"].isin([6,7]))].copy()
        for _,fe in focus.iterrows():
            st.metric(f"Score {int(fe['Umbral'])} SHORT · {int(fe['Horizonte (h)'])}h",f"{fe['Media %']:.3f}%","vs benchmark %+ .3f pp"%fe["Ventaja vs benchmark %"])
        bestedge=focus.sort_values(["Ventaja vs benchmark %","Señales"],ascending=[False,False]).iloc[0]
        st.success(f"Mayor ventaja OOS: Score {int(bestedge['Umbral'])} SHORT a {int(bestedge['Horizonte (h)'])}h → {bestedge['Ventaja vs benchmark %']:.3f} puntos porcentuales.")
        st.info("V1.15.0 congela las cuatro hipótesis antes de mirar el resultado: no selecciona un nuevo TP/SL ni recalibra la señal. Si una ventaja es real, debería superar el benchmark y mostrar estabilidad en varios bloques OOS.")
        

        st.markdown("#### 🛡️ Robustez frente a outliers — OOS")
        st.caption("Quitamos las mejores señales del resultado, sin recalibrar nada. Si la media sigue siendo positiva, el efecto depende menos de unas pocas operaciones extremas.")
        rob=st.session_state["tf_robust_oos"].copy()
        st.dataframe(rob.round(3),use_container_width=True,hide_index=True)
        st.markdown("##### 📈 Estabilidad por bloques — SHORT")
        rb=st.session_state["tf_robust_blocks_oos"].copy()
        st.dataframe(rb.round(3),use_container_width=True,hide_index=True)
        st.caption("Los 4 bloques son cronológicos dentro de cada hipótesis. Positivo = media del bloque > 0.")
        for _,rr in rob.iterrows():
            blocks=rb[(rb["Umbral"]==rr["Umbral"])&(rb["Horizonte (h)"]==rr["Horizonte (h)"])]
            pos=int(blocks["Positivo"].sum()) if not blocks.empty else 0
            worst=float(blocks["Media %"].min()) if not blocks.empty else 0.
            st.metric(f"Score {int(rr['Umbral'])} SHORT · {int(rr['Horizonte (h)'])}h",f"{rr['Media original %']:.3f}% → {rr['Sin top 10% %']:.3f}%","%d/4 bloques positivos · peor %.3f%%"%(pos,worst))
        st.info("La eliminación de outliers es una prueba de sensibilidad, no una nueva optimización. Un edge sano no debería desaparecer por completo al retirar una pequeña fracción de los mejores resultados.")
        

        st.markdown("#### 🧭 V1.15.0 — Validación temporal rigurosa")
        st.caption("Hipótesis fijadas de antemano: Score 6/7, LONG/SHORT y 24/48h. El moving-block bootstrap conserva dependencia local entre retornos; el benchmark aleatorio se muestrea dentro de bloques cronológicos equivalentes.")
        tv=st.session_state["tf_temporal_oos"].copy()
        st.dataframe(tv.round(3),use_container_width=True,hide_index=True)
        st.caption("El intervalo MBB es más prudente que remuestrear operaciones independientes, pero no elimina por sí solo todos los sesgos ni sustituye una validación externa independiente.")
        st.markdown("##### Walk-forward descriptivo — cuatro segmentos OOS")
        tw=st.session_state["tf_walk_oos"].copy()
        st.dataframe(tw.round(3),use_container_width=True,hide_index=True)
        for _,trv in tv[(tv["Dirección"]=="SHORT")&(tv["Horizonte (h)"].isin([24,48]))].iterrows():
            seg=tw[(tw["Umbral"]==trv["Umbral"])&(tw["Dirección"]==trv["Dirección"])&(tw["Horizonte (h)"]==trv["Horizonte (h)"])]
            positives=int(seg["Media positiva"].sum()) if not seg.empty else 0
            st.metric(f"Score {int(trv['Umbral'])} SHORT · {int(trv['Horizonte (h)'])}h",
                      f"{trv['Media %']:.3f}%",
                      f"MBB IC95 {trv['MBB IC95 inferior %']:.3f} a {trv['MBB IC95 superior %']:.3f}% · {positives}/4 bloques positivos")
        st.info("V1.15.0 no cambia TP/SL ni convierte automáticamente una hipótesis en estrategia. Solo se considerará avanzar si la ventaja persiste con incertidumbre temporal y en segmentos posteriores.")
        
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
        export_cfg=comp.copy(); export_cfg["Tipo registro"]="Configuración desarrollo"; export_cfg["Segmento"]="Desarrollo"
        export_oos=st.session_state["tf_oos"].copy(); export_oos["Tipo registro"]="Resultado OOS"; export_oos["Segmento"]="OOS"
        export_tr=tr.copy(); export_tr["Tipo registro"]="Operación OOS"; export_tr["Segmento"]="OOS"
        export_hc=st.session_state["tf_horizon_close_oos"].copy(); export_hc["Tipo registro"]="Horizonte cierre-señal"; export_hc["Segmento"]="OOS"
        export_he=st.session_state["tf_horizon_exec_oos"].copy(); export_he["Tipo registro"]="Horizonte next-open"; export_he["Segmento"]="OOS"
        export_rd=st.session_state["tf_regime_dev"].copy(); export_rd["Tipo registro"]="Régimen"; export_rd["Segmento"]="Desarrollo"
        export_ro=st.session_state["tf_regime_oos"].copy(); export_ro["Tipo registro"]="Régimen"; export_ro["Segmento"]="OOS"
        export_rhd=st.session_state["tf_regime_horizon_dev"].copy(); export_rhd["Tipo registro"]="Régimen + horizonte"; export_rhd["Segmento"]="Desarrollo"
        export_rho=st.session_state["tf_regime_horizon_oos"].copy(); export_rho["Tipo registro"]="Régimen + horizonte"; export_rho["Segmento"]="OOS"
        export_edge=st.session_state["tf_edge_oos"].copy(); export_edge["Tipo registro"]="Validación estadística edge"; export_edge["Segmento"]="OOS"
        export_blocks=st.session_state["tf_edge_blocks_oos"].copy(); export_blocks["Tipo registro"]="Estabilidad temporal"; export_blocks["Segmento"]="OOS"
        export_rob=st.session_state["tf_robust_oos"].copy(); export_rob["Tipo registro"]="Robustez outliers"; export_rob["Segmento"]="OOS"
        export_rob_blocks=st.session_state["tf_robust_blocks_oos"].copy(); export_rob_blocks["Tipo registro"]="Estabilidad bloques robustez"; export_rob_blocks["Segmento"]="OOS"
        export_tv=st.session_state["tf_temporal_oos"].copy(); export_tv["Tipo registro"]="Validación temporal MBB + benchmark aleatorio auditado + bootstrap corregido"; export_tv["Segmento"]="OOS"
        export_tw=st.session_state["tf_walk_oos"].copy(); export_tw["Tipo registro"]="Bloques cronológicos descriptivos"; export_tw["Segmento"]="OOS"
        export_wf=st.session_state["tf_walk_forward"].copy(); export_wf["Tipo registro"]="Walk-forward expanding window V1.15.0"; export_wf["Segmento"]="OOS secuencial"
        export_vd=st.session_state["tf_variant_diagnostic"].copy(); export_vd["Tipo registro"]="Diagnóstico estadístico variantes V1.15.0"; export_vd["Segmento"]="OOS por variante"
        export_all=pd.concat([export_cfg,export_oos,export_tr,export_hc,export_he,export_rd,export_ro,export_rhd,export_rho,export_edge,export_blocks,export_rob,export_rob_blocks,export_tv,export_tw,export_wf,export_vd],ignore_index=True,sort=False)
        st.download_button("Descargar diagnóstico completo V1.16.0",export_all.to_csv(index=False).encode("utf-8"),file_name="btc_ai_trader_predictive_edge_v1_16_1.csv",mime="text/csv")
    st.divider()
    st.caption("V1.16.1: incorpora una prueba confirmatoria única sobre el último 20% cronológico reservado. Test bruto sin costes; el benchmark iguala recuento mensual, horizonte y no solapamiento. La validación no garantiza resultados futuros.")

render()
