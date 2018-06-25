import sim
import json
import random
import typing
import pandas as pd
from sys import argv, stderr
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
        resp_dict = json.loads(resp_str)
        results = resp_dict['results']
        if results:
            return pd.Series(results[0])
        else:
            return None

    def get_app(self, app_id) -> pd.Series:
        if app_id in self.cache.index:
            return self.cache.loc[app_id]
        else:
            return self.fetch_app(app_id)

    def get_validate_apps(self, current_app_id, eligible_apps_ids):
        current_app = self.get_app(current_app_id)
        if current_app is None:
            print("Installed app %s not found, aborting" % current_app_id, file=stderr)
            exit(-1)
        eligible_apps = list(map(self.get_app, eligible_apps_ids))
        err_apps_ids = {eligible_apps_ids[i] for i in range(len(eligible_apps_ids)) if eligible_apps[i] is None}
        if err_apps_ids:
            print("Eligible apps %s not found, ignoring" % ' '.join(err_apps_ids), file=stderr)
            eligible_apps_ids = [x for x in eligible_apps_ids if x not in err_apps_ids]
            eligible_apps = list(filter(lambda x: x is not None, eligible_apps))
        return current_app, eligible_apps, eligible_apps_ids

    def app_recommend(self, current_app_id: str, eligible_apps_ids: typing.List[str]) -> typing.Dict[str, float]:
        current_app, eligible_apps, eligible_apps_ids = self.get_validate_apps(current_app_id, eligible_apps_ids)
        current_app_text = current_app.description
        eligible_apps_df = pd.DataFrame(eligible_apps, eligible_apps_ids)
        eligible_apps_texts = eligible_apps_df.description
        sim_scores = self.sim.sim(current_app_text, eligible_apps_texts)
        scores_dict = dict(zip(eligible_apps_ids, sim_scores))
        return scores_dict


if __name__ == '__main__':
    rec = Recommend('word_to_idx.pickle', 'embed.npy', 'apps.json.gz')
    cache = rec.cache
    curr = argv[1] if len(argv) > 1 else random.choice(cache.index)
    elig = argv[2:] if len(argv) > 2 else random.choices(cache.index, k=5)
    res = rec.app_recommend(curr, elig)
    series = pd.Series(res)
    print(series.to_csv())
