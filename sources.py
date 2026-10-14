"""Read-only discovery of official tables; never substitutes invented official data."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

SOURCE_URL='https://stat.gov.kz/industries/social-statistics/demography/dynamic-tables/'

def discover():
    try:
        response=requests.get(SOURCE_URL,timeout=(5,15))
        response.raise_for_status()
    except requests.exceptions.ProxyError:
        return {'ok':False,'status':'blocked','message':'Желі проксиі stat.gov.kz сұрауын бұғаттады. Орта баптауларында stat.gov.kz доменіне рұқсат беріп, қайта тексеріңіз.','url':SOURCE_URL,'tables':[]}
    except requests.RequestException:
        return {'ok':False,'status':'unavailable','message':'Ресми сайттан жауап алынбады. Кейінірек қайталап көріңіз немесе ресми кестені импорттаңыз.','url':SOURCE_URL,'tables':[]}
    soup=BeautifulSoup(response.text,'html.parser')
    tables=[]
    for row in soup.select('.divTableRow'):
        cells=row.select('.divTableCell')
        if not cells: continue
        title=cells[0].get_text(' ',strip=True)
        if ('ішкі көші-қон' in title and 'жылдар' in title) or 'орташа жылдық саны' in title:
            for a in row.select('a[href]'):
                if a.get_text(strip=True)!='xlsx': continue
                url=urljoin(SOURCE_URL,a['href'])
                if urlparse(url).scheme=='https' and urlparse(url).hostname=='stat.gov.kz':
                    tables.append({'title':title,'url':url})
    return {'ok':True,'status':'reachable','message':'Ресми бет қолжетімді. Кестенің ішкі көші-қонға және бірдей өңір шекараларына қатысты екенін тексеріп, CSV пішіміне келтіріп импорттаңыз.','url':SOURCE_URL,'tables':tables[:40]}
