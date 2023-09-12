"""utils for file management"""
import os

# import sys
# from inspect import FrameInfo
# from pathlib import Path


def get_project_root_dir() -> str:
    """gets the name of the project directory"""
    path = os.path.dirname(os.path.abspath("__init__.py"))
    pathsplit = path.split("average-reward-reinforcement-learning")
    # print(pathsplit)
    # return pathsplit[0]+'average-reward-reinforcement-learning\\'
    # # Uncomment previous line for windows systems
    return (
        pathsplit[0] + "average-reward-reinforcement-learning"
    )  # comment this line for windows systems
