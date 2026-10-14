import io
import numpy as np
import pandas as pd
import pytest
import requests
from analytics import ROOT, IDS, DataError, validate_panel, weights, moran, features, fit, predict, analyse
from app import app
import sources

@pytest.fixture(scope='module')
def panel():
    return pd.read_csv(ROOT/'data/demo.csv')

@pytest.fixture(scope='module')
def analysis(panel):
    return analyse(panel,'spatial',3,4)


def test_balanced_synthetic_data_explicitly_identified(panel):
    clean=validate_panel(panel)
    assert len(clean)==200
    totals=clean.groupby('year')[['arrivals','departures']].sum()
    assert (totals.arrivals==totals.departures).all()
    response=app.test_client().get('/api/analysis?dataset=demo')
    assert response.status_code==200
    assert response.json['dataset']['kind']=='demo'
    assert 'ресми статистика емес' in response.json['dataset']['source']


@pytest.mark.parametrize('mutation',[
    lambda x:x.drop(columns='population'),
    lambda x:pd.concat([x,x.iloc[[0]]]),
    lambda x:x.drop(index=0),
    lambda x:x.assign(arrivals=-1),
    lambda x:x.assign(population=0),
    lambda x:x.assign(arrivals=np.nan),
    lambda x:x.assign(arrivals=1.5),
    lambda x:x[x.year!=2021],
    lambda x:x.assign(region_id='unknown'),
])
def test_reject_invalid_panel(panel,mutation):
    with pytest.raises(DataError): validate_panel(mutation(panel.copy()))


def test_weights_and_moran_reference():
    w=weights(4)
    assert np.allclose(w.sum(axis=1),1)
    assert np.all(np.diag(w)==0)
    assert np.array_equal(w>0,w.T>0)
    z=np.arange(20,dtype=float)
    centered=z-z.mean()
    expected=(centered@w@centered)/(centered@centered)
    result=moran(z,w)
    assert result['i']==pytest.approx(expected)
    assert 0<result['p']<=1
    assert result==moran(z,w)
    assert moran(np.ones(20),w)['i'] is None


def test_time_features_do_not_leak_heldout_arrivals(panel):
    w=weights(); original=features(panel,w)
    changed=panel.copy();changed.loc[changed.year==2025,'arrivals']*=3
    modified=features(changed,w)
    np.testing.assert_allclose(original.spatial_lag,modified.spatial_lag)
    train=original[original.year<2025]
    other_train=modified[modified.year<2025]
    m1=fit(train,'spatial');m2=fit(other_train,'spatial')
    np.testing.assert_allclose(m1['result'].params,m2['result'].params)
    # Forecast validation compares the same regions and population offsets.
    predictions=predict(m1,original[original.year==2025])
    assert np.isfinite(predictions).all() and (predictions>0).all()


def test_forecast_and_validation_integrity(analysis):
    assert analysis['holdout_year']==2025
    assert analysis['model']['observations']==180
    assert analysis['model']['train_observations']==160
    assert len(analysis['comparison'])==4
    assert all(c['rmse']>=c['mae']>=0 for c in analysis['comparison'])
    assert [x['year'] for x in analysis['forecast']]==[2026,2027,2028]
    for forecast in analysis['forecast']:
        assert forecast['lower']<forecast['mean']<forecast['upper']
        assert forecast['mean']==pytest.approx(sum(r['mean'] for r in forecast['regions']))
        assert all(r['lower']>=0 and r['lower']<=r['upper'] for r in forecast['regions'])
    assert analysis['model']['pearson_dispersion']<5


def test_api_validation_and_export():
    client=app.test_client()
    for query in ['model=arbitrary','horizon=0','horizon=6','k=1','k=7','dataset=../../etc/passwd']:
        assert client.get('/api/analysis?'+query).status_code==400
    r=client.get('/api/export?dataset=demo')
    assert r.status_code==200 and 'DEMO' in r.headers['Content-Disposition']
    r=client.post('/api/import',data={'file':(io.BytesIO(b'year,region_id\n2025,astana\n'),'x.csv'),'source':'test','boundaries':'confirmed'})
    assert r.status_code==400 and 'Бағандар жетіспейді' in r.json['error']


def test_official_access_failure_never_reports_data_success(monkeypatch):
    def blocked(*a,**kw):
        raise requests.exceptions.ProxyError('blocked')
    monkeypatch.setattr(sources.requests,'get',blocked)
    result=sources.discover()
    assert result['status']=='blocked'
    assert result['ok'] is False
    assert result['tables']==[]


def test_official_source_integrity_and_fractional_population():
    import hashlib
    import json
    from scripts.import_official import SOURCES,parse_workbook
    meta=json.loads((ROOT/'data/official-metadata.json').read_text())
    frame=validate_panel(pd.read_csv(ROOT/'data/official.csv'))
    assert len(frame)==200 and frame.year.min()==2016 and frame.year.max()==2025
    assert (frame.population % 1 != 0).any() # annual averages must not be rounded away
    assert hashlib.sha256((ROOT/'data/official.csv').read_bytes()).hexdigest()==meta['sha256']
    for source in meta['sources']:
        assert hashlib.sha256((ROOT/source['file']).read_bytes()).hexdigest()==source['sha256']
        values,totals,metadata,sheet=parse_workbook(ROOT/source['file'],source['metric'])
        assert 'қайта есептелді' in metadata['Ескертпе']
        for year in meta['years']:
            data=frame[frame.year==year].set_index('region_id')
            metric=source['metric']
            for region in IDS:
                observed=data.loc[region,'arrivals']-data.loc[region,'departures'] if metric=='balance' else data.loc[region,metric]
                assert observed==values[region][year]
            computed=(data.arrivals-data.departures).sum() if metric=='balance' else data[metric].sum()
            assert computed==totals[year]
    result=app.test_client().get('/api/analysis')
    assert result.status_code==200 and result.json['dataset']['kind']=='official'
    assert result.json['annual'][-1]['arrivals']==1715597
    assert result.json['annual'][-1]['population']==20391610.5


def test_source_discovery_includes_official_table_links(monkeypatch):
    class Reply:
        text='<div class="divTableRow"><div class="divTableCell">Барлық ағындар бойынша ішкі көші-қон - келуі (жылдар бойынша)</div><a href="/api/iblock/element/449811/file/kk/">xlsx</a></div>'
        def raise_for_status(self): pass
    monkeypatch.setattr(sources.requests,'get',lambda *args,**kwargs:Reply())
    result=sources.discover()
    assert result['ok'] is True
    assert result['tables'][0]['url']=='https://stat.gov.kz/api/iblock/element/449811/file/kk/'
