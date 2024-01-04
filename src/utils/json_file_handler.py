import dataclasses
from abc import ABC
from pathlib import Path

import json

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class JsonFileWriter:

    def __init__(self, file_path: Path) -> None:
        self.path = file_path

    def save_as_json(self, obj):

        jsonstr = json.dumps(obj, cls=EnhancedJSONEncoder)

        with open(self.path, "w+") as writer:
            writer.write(jsonstr)
            writer.close()

class FileLoader(ABC):

    def __init__(self, inputfile: Path):

        if inputfile == "":
            raise ValueError("inputfile is empty")

        self.checkFileExists(inputfile)

        self.inputfile = inputfile

    def checkFileExists(self, inputfile: Path):

        if not inputfile.exists():
            message = f"{inputfile} does not exist"
            #LoggingManager.get_default_logger().info(message)
            raise Exception(message)


class JsonFileLoader(FileLoader):

    def loadJsonFile(self) -> dict:
        with open(self.inputfile) as json_file:
            config_dict = json.load(json_file)

        return config_dict