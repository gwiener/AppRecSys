import sim
import json
import random
import typing
import pandas as pd
from urllib import request


class Recommend(object):
    def __init__(self, idx_file, emb_file, cache_file=''):
        self.sim = sim.EmbedHistSimilarity(idx_file, emb_file)
        self.cache: pd.DataFrame = None
        if cache_file:
            self.cache = pd.read_json(cache_file, orient='index', convert_axes=False)

    @staticmethod
    def fetch_app(app_id) -> pd.Series:
        url = "http://itunes.apple.com/lookup?id=" + str(app_id)
        with request.urlopen(url) as f:
            resp_str = f.read().decode()
        resp_dict = json.loads(resp_str)['results'][0]  # FIXME test this
        return pd.Series(resp_dict)

    def get_app(self, app_id) -> pd.Series:
        if app_id in self.cache.index:
            return self.cache.loc[app_id]
        else:
            return self.fetch_app(app_id)

    def app_recommend(self, current_app_id: str, eligible_apps_ids: typing.List[str]) -> typing.Dict[str, float]:
        current_app = self.get_app(current_app_id)
        current_app_text = current_app.description
        eligible_apps = list(map(self.get_app, eligible_apps_ids))
        eligible_apps_df = pd.DataFrame(eligible_apps, eligible_apps_ids)
        eligible_apps_texts = eligible_apps_df.description
        sim_scores = self.sim.sim(current_app_text, eligible_apps_texts)
        scores_dict = dict(zip(eligible_apps_ids, sim_scores))
        return scores_dict


if __name__ == '__main__':
    rec = Recommend('word_to_idx.pickle', 'embed.npy', 'apps.json.gz')
    cache = rec.cache
    curr = random.choice(cache.index)
    elig = random.choices(cache.index, k=5)
    res = rec.app_recommend(curr, elig)
    series = pd.Series(res)
    print(series.to_csv())
