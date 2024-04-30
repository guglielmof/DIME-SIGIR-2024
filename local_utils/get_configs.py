import configparser
from glob import glob
import os


def get_configs(folder="../properties/", local=False):
    file_content = '[root]\n'

    for filename in filter(os.path.isfile, glob(folder + "**/*", recursive=True)):
        with open(filename, "r") as f:
            file_content += f.read()

    config_parser = configparser.RawConfigParser()
    config_parser.read_string(file_content)
    config_parser = config_parser['root']

    if local:
        config_parser['basepath'] = config_parser['basepath_local']

    return config_parser
