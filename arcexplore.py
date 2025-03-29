import requests
from zipfile import ZipFile
from io import BytesIO
import fnmatch
import json
from typing import Mapping,List, Dict, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from collections import UserDict

arcagi_colours = ['#000000',
                  '#1e93ff',
                  '#F93C31',#'#[1,0.2549019607843137,0.21176470588235294,1], # 2 - Red: #F93C31
                  '#4fcc30', # 3 - Green: #2ECC40
                  '#ffdc00', # 4 - Yellow: #FFDC00
                  '#999999', # 5 - Gray:  #AAAAAA
                  '#e53aa3', # 6 - Fuschia: #F012BE
                  '#ff851b', # 7 - Orange: #FF851B
                  '#87d8f1', # 8 - Teal: #7FDBFF;
                  '#921231'] # 9 - Brown: #870C25
arcaci_colormap = ListedColormap(arcagi_colours)

@dataclass
class TaskGridPair:
    input: np.array
    output: np.array

    @classmethod
    def from_dict(cls, d):
        return TaskGridPair(np.asarray(d["input"]),
                          np.asarray(d["output"]))

    def __repr__(self):
        return f"TaskPair[Input: {self.input.shape} -> Output: {self.output.shape}]"

    def display(self, title=None):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        self._plot_grid(self.input, title="Input",ax=ax1)
        self._plot_grid(self.output, title="Output",ax=ax2)
        if title:
            fig.suptitle(title)
        fig.show(warn=False)

    @staticmethod
    def _plot_grid(grid, title=None,ax = None):
        ax = ax or plt.gca()

        plt.style.use('dark_background')
        ax.pcolor(grid.T,
                   edgecolors='grey',
                   linewidths=2,
                   cmap=arcaci_colormap,
                   vmin=0,
                   vmax=9)
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])

@dataclass
class Task:
    id: str
    train: List[TaskGridPair]
    test: List[TaskGridPair]

    @classmethod
    def from_dict(cls, id,d):
        return Task(id,
                    cls._process_caselist(d["train"]),
                    cls._process_caselist(d["test"]))

    @staticmethod
    def _process_caselist(case_list):
        return [TaskGridPair.from_dict(case_dict) for case_dict in case_list]

    def __repr__(self):
        return f"Task[Training: {len(self.train)} -> Test: {len(self.test)}]"

    def display(self):
        for i,v2 in enumerate(self.train):
            v2.display(f"{self.id} Train[{i}]")
            plt.show()
        for i, v2 in enumerate(self.test):
            v2.display(f"{self.id} Test[{i}]")
            plt.show()

    def data_iter(self):
        for ind, tgp in enumerate(self.train):
            yield (self.id, 'train', ind, tgp)
        for ind, tgp in enumerate(self.test):
            yield (self.id, 'test', ind, tgp)

@dataclass
class TaskSet:
    label: str
    training: Dict[str, Task]
    evaluation: Dict[str, Task]

    @classmethod
    def _glob_to_dict(cls, glob_name, zipfile):
        filelist = fnmatch.filter(zipfile.namelist(), glob_name)
        task_gen = lambda id,fn: Task.from_dict(id,json.load(zipfile.open(fn)))
        return {Path(fn).stem: task_gen(Path(fn).stem,fn) for fn in filelist}

    @classmethod
    def from_zipfile(cls, label: str, zipfile: ZipFile):
        return TaskSet(label,
                       cls._glob_to_dict("*/data/training/*.json", zipfile),
                       cls._glob_to_dict("*/data/evaluation/*.json", zipfile))

    @staticmethod
    def _display_dict(task_dict: Mapping[str, Task]) -> None:
        key_list = list(task_dict.keys())
        options=widgets.Dropdown(options=key_list,value=key_list[0],description='ID')
        return widgets.interactive(lambda x: task_dict[x].display(),
                                   {"manual": True},
                                   x = options)

    def display(self):
        tab_nest = widgets.Tab()
        tab_nest.children = [self._display_dict(self.training),
                             self._display_dict(self.evaluation)]
        tab_nest.titles = ('Training', 'Evaluation')
        return tab_nest

    def data_iter(self):
        for _,tgp in self.training.items():
            for case in tgp.data_iter():
                yield (self.label, 'training') + case
        for _,tgp in self.evaluation.items():
            for case in tgp.data_iter():
                yield (self.label, 'evaluation') + case


class ARCSet(UserDict):
    @classmethod
    def from_links(cls, link_dict):
        arc_set = cls()
        for name, link in link_dict.items():
            zip_data = cls.get_zip(link)
            arc_set.data[name] = TaskSet.from_zipfile(name, zip_data)
        return arc_set

    @staticmethod
    def get_zip(link: str) -> ZipFile:
        response = requests.get(link)
        return ZipFile(BytesIO(response.content))

    def display(self):
        tab_nest = widgets.Tab()
        tab_nest.children = [i.display() for _, i in self.items()]
        tab_nest.titles = [k for k in self.keys()]
        display(tab_nest)

    def data_iter(self):
        for _, v in self.items():
            yield from v.data_iter()

ARC_LINKS = {'ARC-AGI-1': "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip",
             'ARC-AGI-2': "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"}
ARC_DATA = ARCSet.from_links(ARC_LINKS)