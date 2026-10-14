from pathlib import Path
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from analytics import ROOT, REGIONS, REQUIRED, DataError, validate_panel, analyse
from sources import discover

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=5*1024*1024
INSTANCE=ROOT/'instance'
INSTANCE.mkdir(exist_ok=True)


def dataset_path(dataset):
    if dataset=='demo': return ROOT/'data/demo.csv'
    if dataset=='official': return ROOT/'data/official.csv'
    if not re.fullmatch(r'[0-9a-f]{24}',dataset): raise DataError('Дерек жиыны табылмады.')
    path=INSTANCE/(dataset+'.csv')
    if not path.is_file(): raise DataError('Дерек жиыны табылмады. CSV файлын қайта импорттаңыз.')
    return path


def metadata(dataset):
    if dataset=='official':
        return json.loads((ROOT/'data/official-metadata.json').read_text(encoding='utf-8'))
    if dataset=='demo':
        return {'kind':'demo','title':'Көрсетілім деректері','source':'Жасанды деректер · ресми статистика емес',
                'note':'2016–2025 жылдардағы сандар интерфейс пен есептеуді сынау үшін генерацияланған. Диплом қорытындыларына қолдануға болмайды. 20 өңірдің тарихи шекаралары бұл мысалда ескерілмеген.'}
    return json.loads((INSTANCE/(dataset+'.json')).read_text(encoding='utf-8'))


@lru_cache(maxsize=12)
def calculation(dataset,kind,horizon,k):
    frame=pd.read_csv(dataset_path(dataset))
    return analyse(frame,kind,horizon,k)


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/health')
def health():
    return jsonify(status='ok',service='QONYS',version='1.0.0')


@app.get('/api/analysis')
def analysis():
    dataset=request.args.get('dataset','official')
    kind=request.args.get('model','spatial')
    if kind not in ['poisson','nb','spatial']: raise DataError('Белгісіз модель.')
    try:
        horizon=int(request.args.get('horizon','3')); k=int(request.args.get('k','4'))
    except ValueError: raise DataError('Көкжиек пен көршілер саны бүтін болуы керек.')
    if not 1<=horizon<=5 or not 2<=k<=6: raise DataError('Көкжиек 1–5 жыл, көршілер саны 2–6 болуы керек.')
    result=calculation(dataset,kind,horizon,k)
    return jsonify(**result,dataset=metadata(dataset))


@app.post('/api/import')
def import_data():
    file=request.files.get('file')
    if not file or not file.filename.lower().endswith('.csv'): raise DataError('UTF-8 CSV файлын таңдаңыз.')
    if request.form.get('boundaries')!='confirmed':
        raise DataError('Өңір шекаралары мен көші-қон түрінің бірізділігін растаңыз.')
    source=request.form.get('source','').strip()
    if not source or len(source)>500: raise DataError('Дереккөз сипаттамасын енгізіңіз (500 таңбаға дейін).')
    try:
        frame=pd.read_csv(file,encoding='utf-8-sig',sep=None,engine='python')
    except (UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError):
        raise DataError('CSV файлын оқу мүмкін емес. UTF-8 кодтамасын және бөлгіштерін тексеріңіз.')
    frame=validate_panel(frame)
    # Confirm all model fits are possible before persisting a dataset.
    analyse(frame,'spatial',1,4)
    data=frame.to_csv(index=False).encode()
    dataset=hashlib.sha256(data+source.encode()).hexdigest()[:24]
    (INSTANCE/(dataset+'.csv')).write_bytes(data)
    meta={'kind':'uploaded','title':'Импортталған деректер','source':source,
          'note':'Пайдаланушы жүктеген деректер. Ресми түпнұсқаға сәйкестігі автоматты түрде тексерілмеген.',
          'imported_at':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(data).hexdigest()}
    (INSTANCE/(dataset+'.json')).write_text(json.dumps(meta,ensure_ascii=False),encoding='utf-8')
    return jsonify(dataset_id=dataset,metadata=meta,rows=len(frame))


@app.post('/api/source/check')
def source_check():
    return jsonify(discover())


@app.get('/api/export')
def export():
    dataset=request.args.get('dataset','official')
    return send_file(dataset_path(dataset),mimetype='text/csv',as_attachment=True,
                     download_name='qonys-DEMO.csv' if dataset=='demo' else 'qonys-'+dataset+'.csv')


@app.get('/api/provenance')
def provenance():
    return send_file(ROOT/'data/official-metadata.json',mimetype='application/json',
                     as_attachment=True,download_name='qonys-official-provenance.json')


@app.get('/api/template')
def template():
    csv='year,region_id,population,arrivals,departures\n'+'\n'.join(f'2023,{r["id"]},,,' for r in REGIONS)+'\n'
    return send_file(io.BytesIO(csv.encode('utf-8-sig')),mimetype='text/csv',as_attachment=True,download_name='qonys-template.csv')


@app.errorhandler(DataError)
def invalid(error):
    return jsonify(error=str(error)),400


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return jsonify(error='Файл көлемі 5 МБ-тан аспауы керек.'),413


@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    return response


if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','8000')),debug=False)
