import os
import yaml

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

def index_prices(index_name:str):
    return component_prices(index_name, f'^{index_name.upper()}.csv')


def component_prices(index_name:str, component_name:str):
    filename = f'{component_name.upper()}.csv'
    path = os.path.join(config["STOCK_DATA_PATH"], index_name, filename)
    with open(path, 'r') as f:
        lines = f.read().split('\n')[1:]
        lines = filter(lambda line: len(line) > 0, lines)
        lines = [float(x.split(',')[1]) for x in lines]

    return lines

def index_component_names(index_name:str):
    dir = os.path.join(config["STOCK_DATA_PATH"], index_name)
    files = os.listdir(dir)
    files = map(lambda x: x[:-4], files)
    files = filter(lambda x: not x.startswith('^'), files)
    return list(files)





def component_stonk_prices(index_name:str, component_name:str):
    filename = f'{component_name.upper()}.csv'
    path = os.path.join()



if __name__ == '__main__':
    print(index_component_names('gspc'))