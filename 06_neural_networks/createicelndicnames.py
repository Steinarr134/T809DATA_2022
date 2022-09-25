import pandas as pd

with open("data/nafnalisti.csv", 'r') as f:
    data = pd.read_csv(f)

nofn = data.loc[data["Afgreitt"]=="Sam"]["Nafn"]

with open("data/names/ice.txt", 'w+') as f:
    f.write("\n".join(nofn))

