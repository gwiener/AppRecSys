import sim
import json
import random
import typing
import pandas as pd
import configargparse
from os import path
from sys import stderr
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
    parser = configargparse.ArgumentParser(default_config_files=['recommend.conf'])
    parser.add_argument('--emb-file', type=str)
    parser.add_argument('--idx-file', type=str)
    parser.add_argument('--cache-file', type=str)
    parser.add_argument('-k', '--num-rand-eligible', type=int)
    parser.add_argument('app', nargs='?')
    parser.add_argument('eligible_apps', nargs='*')
    args = parser.parse_args()
    if not args.app and not (args.cache_file and path.isfile(args.cache_file)):
        print("No app ids given and no cache file found, aborting", file=stderr)
        exit(-1)
    rec = Recommend(args.idx_file, args.emb_file, args.cache_file)
    cache = rec.cache
    curr = args.app if args.app else random.choice(cache.index)
    elig = args.eligible_apps if args.eligible_apps else random.choices(cache.index, k=args.num_rand_eligible)
    res = rec.app_recommend(curr, elig)
    series = pd.Series(res)
    print(series.to_csv())
