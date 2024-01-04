from typing import Type
from datetime import datetime
from pathlib import Path
import os
from enum import Enum
import inspect
from shutil import move

from .dynamictypeloader import load_type_dynamically_from_fqn
from .utils import get_project_root
from .json_file_handler import JsonFileLoader, JsonFileWriter





class ConfigParser:

    def __init__(self, mode:str, exp_name: str=None, datastructure_module_name: str = "src.utils.datastructures") -> None:
        self.datastructure_module_name = datastructure_module_name
        self.mode = mode # either train or test, relevant for output folder creation
        self.exp_name = exp_name

    def parse_config_from_args(self, args):
        """
        loads a json config from the specified location via --config commandline flag and parses it into a typed object based on the type specified in 'type_name'
        """
        if not isinstance(args, tuple):
            args = args.parse_args()
        config_path = Path(args.config)

        if not isinstance(config_path, Path):
            raise TypeError("'config_path' must be a Path")

        if not config_path.exists():
            raise ValueError(
                f"given config_path '{config_path}' does not exist")

        # this weirdly doesnt work.. torch.cuda.device_count still sees 4 gpus
        # must apparently be set before torch is imported
        # but this setup works for transformerflow wo not sure what happens
        #( had troubles with transformer flwo as well and suddenly it worked evethough i didnt change anything (knowingly))
        # if args.device is not None:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        gpu_id = args.device
        # read in config 
        fileloader = JsonFileLoader(config_path)
        config_dict = fileloader.loadJsonFile()
       
        if self.exp_name is None:
            self.exp_name = config_path.stem
        # if resuming previous model - get this path
        resume = args.resume
        if resume is not None: 
            resume = Path(resume)
            config_path_resume = resume.parent / (self.exp_name + '.json')
            fileloader_resume = JsonFileLoader(config_path_resume)
            config_dict_resume = fileloader_resume.loadJsonFile()
            # check if config has changed:
            # assert config_dict == config_dict_resume, \
            # f'config in resume folder {resume} and given config {config_path} are not the same!'
            if config_dict != config_dict_resume:
                print(f"WARNING: config in resume folder {resume} and given config {config_path} are not the same!")
            if self.mode == "train":
                # overwrite output_save_dir
                output_save_dir = resume.parent

        
        # output save directories depending on testing or training mode
        if self.mode == "test":
            output_save_dir = get_project_root() / Path("testing_output") / self.exp_name # name of config used
            if output_save_dir.exists():
                # move existing one to archive
                
                move_dest = get_project_root() / Path("testing_output") / "archive" / self.exp_name
                i = 1
                while Path(str(move_dest) + f"_{i}").exists():
                    i += 1

                move_dest = Path(str(move_dest) + f"_{i}")
                print(f'folder exists => Moved to {move_dest}')

                move(str(output_save_dir), str(move_dest))

        if self.mode == "viz_embedding":
            output_save_dir = get_project_root() / Path("embeddings_viz") / self.exp_name # name of config used
            if output_save_dir.exists():
                # move existing one to archive
                
                move_dest = get_project_root() / Path("embeddings_viz") / "archive" / self.exp_name
                i = 1
                while Path(str(move_dest) + f"_{i}").exists():
                    i += 1

                move_dest = Path(str(move_dest) + f"_{i}")
                print(f'folder exists => Moved to {move_dest}')

                move(str(output_save_dir), str(move_dest))
            
        if self.mode == "train":
            output_save_dir = get_project_root() / Path("training_output") / self.exp_name # name of config used
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            output_save_dir = output_save_dir / run_id
            

        # create dir
        output_save_dir.mkdir(parents=True, exist_ok=False) # exist - wont happen due to timestamp and moving, but checking here to be safe

        # save config to train_output folder
        if self.mode == "train":
            JsonFileWriter(output_save_dir /(self.exp_name + ".json")).save_as_json(config_dict)
        
        # make subfolder for figures if in test mode
        fig_save_dir = None
        if self.mode == "test":
            # make figure folder:
            fig_save_dir = output_save_dir / "figures"
            fig_save_dir.mkdir(parents=False, exist_ok=False)
        if self.mode == "viz_embedding":
            # make figure folder:
            fig_save_dir = output_save_dir 


       
        # read in config
        fileloader = JsonFileLoader(config_path)
        config_dict = fileloader.loadJsonFile()

        # update config with path to resum model/or None and to training save dir
        config_dict.update({"resume_path": resume})
        config_dict.update({"output_save_dir": output_save_dir})
        config_dict.update({"config_name": self.exp_name})
        config_dict.update({"figures_save_dir": fig_save_dir})
        config_dict.update({"gpu_id": gpu_id})

        # start parsing
        return self.parse_config(config_dict)

    def parse_config_from_file(self, config_path: Path, gpu_id: str='0'):
        """
        loads a json config from the specified location and parses it into a typed object based on the type specified in 'type_name'
        """

        if not isinstance(config_path, Path):
            raise TypeError("'config_path' must be a Path")

        if not config_path.exists():
            raise ValueError(
                f"given config_path '{config_path}' does not exist")

        # read in config 
        fileloader = JsonFileLoader(config_path)
        config_dict = fileloader.loadJsonFile()
       

        if self.exp_name is None:
            self.exp_name = config_path.stem

        output_save_dir = None
        # output save directories depending on testing or training mode
        if self.mode == "test":
            output_save_dir = get_project_root() / Path("testing_output") / self.exp_name # name of config used
            if output_save_dir.exists():
                # move existing one to archive
                
                move_dest = get_project_root() / Path("testing_output") / "archive" / self.exp_name
                i = 1
                while Path(str(move_dest) + f"_{i}").exists():
                    i += 1

                move_dest = Path(str(move_dest) + f"_{i}")
                print(f'folder exists => Moved to {move_dest}')

                move(str(output_save_dir), str(move_dest))
        if self.mode == "viz_embedding":
            output_save_dir = get_project_root() / Path("embeddings_viz") / self.exp_name # name of config used
            if output_save_dir.exists():
                # move existing one to archive
                
                move_dest = get_project_root() / Path("embeddings_viz") / "archive" / self.exp_name
                i = 1
                while Path(str(move_dest) + f"_{i}").exists():
                    i += 1

                move_dest = Path(str(move_dest) + f"_{i}")
                print(f'folder exists => Moved to {move_dest}')

                move(str(output_save_dir), str(move_dest))
            
        if self.mode == "train":
            output_save_dir = get_project_root() / Path("training_output") / self.exp_name # name of config used
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            output_save_dir = output_save_dir / run_id
            

        # create dir
        if output_save_dir is not None:
            output_save_dir.mkdir(parents=True, exist_ok=False) # exist - wont happen due to timestamp and moving, but checking here to be safe

        # save config to train_output folder
        if self.mode == "train":
            JsonFileWriter(output_save_dir /(self.exp_name + ".json")).save_as_json(config_dict)
        
        # make subfolder for figures if in test mode
        fig_save_dir = None
        if self.mode == "test":
            # make figure folder:
            fig_save_dir = output_save_dir / "figures"
            fig_save_dir.mkdir(parents=False, exist_ok=False)
        if self.mode == "viz_embedding":
            # make figure folder:
            fig_save_dir = output_save_dir 
       
        # read in config
        fileloader = JsonFileLoader(config_path)
        config_dict = fileloader.loadJsonFile()

        # update config with path to resum model/or None and to training save dir
        config_dict.update({"resume_path": None}) # gets changed in e.g. batch_train_test.py
        config_dict.update({"output_save_dir": output_save_dir})
        config_dict.update({"config_name": self.exp_name})
        config_dict.update({"figures_save_dir": fig_save_dir})
        config_dict.update({"gpu_id": gpu_id})

        # start parsing
        return self.parse_config(config_dict)

    def parse_config(self, config_dict: dict):
        """
        parse a config dict into a typed object based in on the type specified in 'type_name'
        """

        if not isinstance(config_dict, dict):
            raise TypeError("'config_dict' must be a dict")

        if "type_name" not in config_dict.keys():
            raise ValueError("'type_name' must be specified")

        current_type = load_type_dynamically_from_fqn(config_dict["type_name"])
        del config_dict["type_name"]  # should not be parsed

        return self.parse_dict_into_typed_object(config_dict, current_type)

    def parse_dict_into_typed_object(self, dict_data: dict, current_type: Type):
        """
        recursively converts a dict into the given dataclass type
        """

        if not isinstance(dict_data, dict) or not hasattr(current_type, "__dataclass_fields__"):
            return dict_data

        result_dict = {}
        current_fields = current_type.__dataclass_fields__

        for k, v in dict_data.items():

            if k not in current_fields:
                raise TypeError(
                    f"unkown field name '{k}' for type '{current_type.__name__}'")

            field = current_fields[k]

            if v is None:
                result_dict[k] = None

            elif inspect.isclass(field.type) and issubclass(field.type, Enum):  # is Enum
                try:
                    result_dict[k] = field.type[v]
                except Exception as ex:
                    raise ValueError(
                        f"value '{v}' is not valid for enum of type '{field.type}' ") from ex

            # complex type from the specificed module
            elif field.type.__module__.startswith(self.datastructure_module_name):
                result_dict[k] = self.parse_dict_into_typed_object(v, field.type)
            elif hasattr(field.type, "_name") and field.type._name == "List":

                if not isinstance(v, list):
                    raise TypeError(f"'{k}' must be a list.")

                list_element_type = field.type.__args__[0]

                parsed_list = []
                for el in v:
                    parsed_list.append(
                        self.parse_dict_into_typed_object(el, list_element_type))

                result_dict[k] = parsed_list
            elif hasattr(field.type, "_name") and field.type._name == "Dict":
                result_dict[k] = dict(v)
            else:  # every other type
                result_dict[k] = field.type(v)

        return current_type(**result_dict)


