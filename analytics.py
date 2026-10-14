"""Count regression and explicitly defined spatial diagnostics for regional panels."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
REGIONS = json.loads((ROOT/'data/regions.json').read_text(encoding='utf-8'))
IDS = [r['id'] for r in REGIONS]
REQUIRED = ['year', 'region_id', 'population', 'arrivals', 'departures']
MODEL_NAMES = {'poisson':'Пуассон', 'nb':'Теріс биномдық (NB2)', 'spatial':'Кеңістіктік NB2'}

class DataError(ValueError):
    pass


def validate_panel(frame):
    """Require a balanced panel and stable territorial definitions supplied by the user."""
    missing = set(REQUIRED)-set(frame.columns)
    if missing:
        raise DataError('Бағандар жетіспейді: '+', '.join(sorted(missing)))
    df = frame[REQUIRED].copy()
    if len(df) > 2000:
        raise DataError('2000 жолдан артық дерек қабылданбайды.')
    if df.isna().any().any():
        raise DataError('Бос мәндер бар. Бос ұяшықтарды нөлмен автоматты алмастырмаңыз.')
    df['region_id'] = df['region_id'].astype(str).str.strip()
    unknown = set(df.region_id)-set(IDS)
    if unknown:
        raise DataError('Белгісіз region_id: '+', '.join(sorted(unknown)))
    for col in ['year','population','arrivals','departures']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all():
            raise DataError(col+': тек шекті сандар болуы керек.')
        if col != 'population' and (df[col] % 1 != 0).any():
            raise DataError(col+': тек бүтін сандар болуы керек.')
    if (df.population <= 0).any() or (df.population > 100_000_000).any():
        raise DataError('population мәндері 1–100 000 000 аралығында болуы керек.')
    if (df[['arrivals','departures']] < 0).any().any() or (df[['arrivals','departures']] > 100_000_000).any().any():
        raise DataError('Көші-қон саны 0–100 000 000 аралығында болуы керек.')
    if df.duplicated(['year','region_id']).any():
        raise DataError('Бір жыл мен өңір үшін қайталанған жолдар бар.')
    years = sorted(df.year.astype(int).unique().tolist())
    if len(years)<4 or years != list(range(years[0],years[-1]+1)):
        raise DataError('Кемінде 4 қатарлас толық жыл қажет; уақыт қатарында бос жыл болмауы керек.')
    if years[0]<1991 or years[-1]>2100:
        raise DataError('Жылдар 1991–2100 аралығында болуы керек.')
    for year,group in df.groupby('year'):
        if set(group.region_id)!=set(IDS):
            raise DataError(f'{int(year)}: барлық 20 өңір қажет. Әкімшілік шекараларды біріздендіріңіз.')
    for col in ['year','arrivals','departures']:
        df[col]=df[col].astype('int64')
    df['population']=df.population.astype(float)
    if (df.groupby('region_id').arrivals.sum()==0).any():
        raise DataError('Барлық жылдары келушілер саны нөл болатын өңірге бұл модельді бағалау мүмкін емес.')
    return df.sort_values(['year','region_id']).reset_index(drop=True)


def weights(k=4):
    """Symmetrised k-nearest representative points, haversine distance, row standardised."""
    coords=np.radians([[r['lat'],r['lon']] for r in REGIONS])
    lat=coords[:,0]; lon=coords[:,1]
    a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    distances=2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))
    np.fill_diagonal(distances,np.inf)
    w=np.zeros((len(IDS),len(IDS)))
    for i in range(len(IDS)):
        w[i,np.argsort(distances[i])[:k]]=1
    w=np.maximum(w,w.T)
    return w/w.sum(axis=1,keepdims=True)


def moran(values,w,permutations=999,seed=417):
    values=np.asarray(values,dtype=float)
    z=values-values.mean()
    denominator=float(z@z)
    if denominator<1e-12:
        return {'i':None,'p':None,'expected':-1/(len(z)-1),'permutations':permutations,'points':[]}
    n=len(z); scale=n/w.sum()
    value=float(scale*(z@w@z)/denominator)
    rng=np.random.default_rng(seed)
    null=np.array([scale*(p@w@p)/denominator for p in (rng.permutation(z) for _ in range(permutations))])
    expected=-1/(n-1)
    p=float((1+np.count_nonzero(np.abs(null-expected)>=abs(value-expected)))/(permutations+1))
    standard=z/z.std()
    return {'i':value,'p':p,'expected':expected,'permutations':permutations,
            'points':[{'region_id':rid,'x':float(x),'y':float(y)} for rid,x,y in zip(IDS,standard,w@standard)]}


def features(df,w):
    result=df.copy()
    result['spatial_lag']=np.nan
    result['trend']=result.year-int(df.year.min())
    for year in sorted(df.year.unique())[1:]:
        previous=df[df.year==year-1].set_index('region_id').loc[IDS]
        lag=dict(zip(IDS,np.log1p(w@(previous.arrivals.to_numpy()/previous.population.to_numpy()*1000))))
        mask=result.year==year
        result.loc[mask,'spatial_lag']=result.loc[mask,'region_id'].map(lag)
    return result.dropna().reset_index(drop=True)


def design(frame,spatial=False):
    x=pd.DataFrame({'const':np.ones(len(frame)), 'trend':frame.trend.to_numpy()},index=frame.index)
    for region in IDS[1:]:
        x['region_'+region]=(frame.region_id==region).astype(float)
    if spatial:
        x['spatial_lag']=frame.spatial_lag.to_numpy()
    return x.astype(float)


def fit(frame,kind):
    x=design(frame,kind=='spatial')
    offset=np.log(frame.population.to_numpy())
    if len(frame)<=len(x.columns)+2 or np.linalg.matrix_rank(x)<len(x.columns):
        raise DataError('Модель үшін тәуелсіз бақылаулар жеткіліксіз немесе белгілер коллинеарлы.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        baseline=sm.GLM(frame.arrivals,x,family=sm.families.Poisson(),offset=offset).fit(maxiter=200)
        mu=np.asarray(baseline.fittedvalues)
        alpha=float(np.clip(np.sum((frame.arrivals.to_numpy()-mu)**2-mu)/np.sum(mu**2),1e-8,10))
        family=sm.families.Poisson() if kind=='poisson' else sm.families.NegativeBinomial(alpha=alpha)
        fitted=sm.GLM(frame.arrivals,x,family=family,offset=offset).fit(maxiter=200,cov_type='HC1')
    if not fitted.converged or not np.isfinite(fitted.params).all():
        raise DataError('Модель жинақталмады. Деректер мен шектен тыс мәндерді тексеріңіз.')
    return {'result':fitted,'alpha':0. if kind=='poisson' else alpha,'kind':kind}


def predict(model,frame):
    return np.asarray(model['result'].predict(design(frame,model['kind']=='spatial'),offset=np.log(frame.population.to_numpy())),dtype=float)


def metrics(actual,predicted):
    actual=np.asarray(actual); predicted=np.asarray(predicted)
    return {'mae':float(np.mean(np.abs(actual-predicted))),
            'rmse':float(np.sqrt(np.mean((actual-predicted)**2))),
            'wape':float(np.sum(np.abs(actual-predicted))/max(np.sum(actual),1)*100)}


def analyse(df,model_kind='spatial',horizon=3,k=4):
    df=validate_panel(df)
    w=weights(k)
    panel=features(df,w)
    last=int(df.year.max()); first=int(df.year.min())
    train=panel[panel.year<last]
    holdout=panel[panel.year==last]
    comparisons=[]
    for kind in MODEL_NAMES:
        trained=fit(train,kind)
        errors=metrics(holdout.arrivals,predict(trained,holdout))
        comparisons.append(dict(id=kind,name=MODEL_NAMES[kind],**errors))
    previous=df[df.year==last-1].set_index('region_id').arrivals
    comparisons.append(dict(id='naive',name='Базалық: өткен жыл',**metrics(holdout.arrivals,holdout.region_id.map(previous))))
    fitted=fit(panel,model_kind)
    result=fitted['result']; mu=predict(fitted,panel)
    panel=panel.copy(); panel['fitted']=mu
    dispersion=float(result.pearson_chi2/result.df_resid)
    latest=df[df.year==last].set_index('region_id').loc[IDS]
    latest_fit=panel[panel.year==last].set_index('region_id').loc[IDS]
    residual=(latest_fit.arrivals-latest_fit.fitted)/np.sqrt(latest_fit.fitted+fitted['alpha']*latest_fit.fitted**2)
    spatial={str(int(year)):moran((group.set_index('region_id').loc[IDS].arrivals/group.set_index('region_id').loc[IDS].population*1000).to_numpy(),w) for year,group in df.groupby('year')}
    spatial['residual']=moran(residual.to_numpy(),w)
    # Conditional predictive simulations, fixed parameters, recursive spatial lag.
    # Population is held at last observed level: a clearly identified scenario.
    rng=np.random.default_rng(2841)
    draws=1200
    history=latest.arrivals.to_numpy(dtype=float)
    trajectories=np.tile(history,(draws,1))
    current_mean=history.copy()
    future=[]
    for step in range(1,horizon+1):
        frame=latest.reset_index()[['region_id','population']].copy()
        frame['trend']=last+step-first
        frame['spatial_lag']=np.log1p(w@(current_mean/latest.population.to_numpy()*1000))
        current_mean=predict(fitted,frame)
        simulated_mu=np.tile(current_mean,(draws,1))
        if model_kind=='spatial':
            simulation_lag=np.log1p((trajectories/latest.population.to_numpy()*1000)@w.T)
            simulated_mu*=np.exp(result.params['spatial_lag']*(simulation_lag-frame.spatial_lag.to_numpy()))
        if not np.isfinite(simulated_mu).all() or simulated_mu.max()>1e9:
            raise DataError('Болжам тұрақсыз. Қысқа көкжиек немесе басқа модель таңдаңыз.')
        alpha=fitted['alpha']
        if alpha>1e-7:
            trajectories=rng.negative_binomial(1/alpha,1/(1+alpha*simulated_mu)).astype(float)
        else:
            trajectories=rng.poisson(simulated_mu).astype(float)
        totals=trajectories.sum(axis=1)
        future.append({'year':last+step,'mean':float(trajectories.sum(axis=1).mean()),
                       'lower':float(np.quantile(totals,.025)),'upper':float(np.quantile(totals,.975)),
                       'regions':[{'region_id':rid,'mean':float(trajectories[:,i].mean()),
                       'lower':float(np.quantile(trajectories[:,i],.025)),'upper':float(np.quantile(trajectories[:,i],.975))} for i,rid in enumerate(IDS)]})
    coefficients=[]
    ci=result.conf_int()
    names={'const':'Тұрақты мүше','trend':'Уақыт тренді','spatial_lag':'Көрші өңірлердің өткен жылғы келу қарқыны (log1p)'}
    for key in ['const','trend','spatial_lag']:
        if key not in result.params: continue
        coefficients.append({'term':key,'name':names[key],'beta':float(result.params[key]),
            'se':float(result.bse[key]),'lower':float(ci.loc[key,0]),'upper':float(ci.loc[key,1]),'p':float(result.pvalues[key])})
    annual=df.groupby('year').agg(arrivals=('arrivals','sum'),departures=('departures','sum'),population=('population','sum')).reset_index()
    annual['rate']=annual.arrivals/annual.population*1000
    imbalance=[int(y) for y,g in df.groupby('year') if int(g.arrivals.sum())!=int(g.departures.sum())]
    notes=[]
    if imbalance: notes.append('Ұлттық келу мен кету сомалары тең емес: '+', '.join(map(str,imbalance))+'. Деректердің қамту аясын тексеріңіз.')
    if len(df.year.unique())<6: notes.append('Жыл саны аз: модель бағалары мен ұзақ мерзімді болжам тұрақсыз болуы мүмкін.')
    return {'annual':annual.to_dict('records'),'rows':df.to_dict('records'),
            'years':sorted(int(y) for y in df.year.unique()),'regions':REGIONS,
            'forecast':future,'comparison':comparisons,'holdout_year':last,
            'model':{'id':model_kind,'name':MODEL_NAMES[model_kind],'alpha':fitted['alpha'],
                     'pearson_dispersion':dispersion,'observations':len(panel),'train_observations':len(train),
                     'coefficients':coefficients,'region_effects':len(IDS)-1},
            'spatial':spatial,'weights':w.tolist(),'k':k,'notes':notes}
