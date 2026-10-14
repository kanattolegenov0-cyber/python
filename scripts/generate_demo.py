"""Generate explicitly synthetic, balanced migration counts; never official statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def generate():
    regions = json.loads((ROOT/'data/regions.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(731)
    population = rng.integers(400_000, 1_500_000, len(regions)).astype(float)
    population[-3:] = [1_050_000, 1_700_000, 950_000]
    attraction = rng.uniform(.7, 1.15, len(regions))
    attraction[-3:] = [3.0, 2.5, 1.5]
    rows = []
    for year in range(2016, 2026):
        total = round(360_000 * 1.047**(year-2016))
        inflow = population * attraction * rng.lognormal(0, .11, len(regions))
        outflow = population * rng.lognormal(0, .09, len(regions))
        arrivals = rng.multinomial(total, inflow/inflow.sum())
        departures = rng.multinomial(total, outflow/outflow.sum())
        for i, region in enumerate(regions):
            rows.append(dict(year=year, region_id=region['id'], population=round(population[i]),
                             arrivals=int(arrivals[i]), departures=int(departures[i])))
        population = np.maximum(100_000, population*1.008 + arrivals-departures)
    pd.DataFrame(rows).to_csv(ROOT/'data/demo.csv', index=False)
if __name__ == '__main__':
    generate()
