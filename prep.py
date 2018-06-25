import json
import tqdm
from sys import argv
from glob import glob

path = argv[1] or '*.json'  # "tmp/*.json" in the original settings
out = argv[2] or 'apps.json'  # was gzipped later in the original settings
dedup_dict = {}
errors = []
for name in tqdm.tqdm(glob(path)):
    with open(name) as f:
        json_str = f.read().strip()
        if json_str:
            try:
                cont = json.loads(json_str)
                for app in cont['results']:
                    tid = app['trackId']
                    del app['trackId']
                    dedup_dict[tid] = app
            except:
                errors.append(name)
                continue
if errors:
    print("Found %d errors" % len(errors))
    print(*errors, sep=',')
print("Writing %d entries" % len(dedup_dict))
with open(out, 'w') as f:
    json.dump(dedup_dict, f)
