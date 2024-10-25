import json

data = {}

with open("items.json", "r") as f:
    data = json.load(f)

lengths = {}

for key in data.keys():
    timeseries = data[key]
    length = len(timeseries)
    if length in lengths.keys():
        lengths[length] += 1
    else:
        lengths[length] = 1

for l in lengths.keys():
    print(f"length {l}: {lengths[l]}")

l = int(input("see length: "))

for key in data.keys():
    ts = data[key]
    length = len(ts)
    if l == length:
        print(ts)
