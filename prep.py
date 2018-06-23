import json
import tqdm
from glob import glob

dedup_dict = {}
errors = []
for name in tqdm.tqdm(glob("tmp/*.json")):
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
with open('apps.json', 'w') as f:
    json.dump(dedup_dict, f)
