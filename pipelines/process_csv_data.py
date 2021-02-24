import os
import yaml

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)

def process_file(file, inpath, outpath):
    with open(os.path.join(inpath, file)) as f:
        csv = f.read()

    idx = 4 if file.startswith('sx5e') else 5

    # this uses adjusted close
    prices = [line.split(',')[idx] for line in csv.split('\n')]
    prices = prices[1:]

    last = prices[1]
    corrected = []
    for price in prices:
        price = price.strip()
        if price == 'null':
            corrected.append(last)
        else:
            corrected.append(price)
            last = price

    with open(os.path.join(outpath, file), 'w') as f:
        f.write(','.join(corrected))




if __name__ == '__main__':
    for file in os.listdir(config["RAW_DATA_PATH"]):
        process_file(file, config["RAW_DATA_PATH"], config["CLEAN_DATA_PATH"])
        print('processed file: ', file)
