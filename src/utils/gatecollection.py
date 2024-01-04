import pandas as pd
from typing import List


class GateCollection:
    """
    GateCollection holds translations from different occuring gate names to the standarised names.
    """

    GATE_BLAST34 = "blast34"
    GATE_BLAST = "blast"
    GATE_BLASTEN = "blasten"
    GATE_BLASTOTHER = "blastother"
    GATE_ADENOMINATOR = "adenominator"
    GATE_INTACT = "intact"
    GATE_CD7pos = "cd7+"
    GATE_BERMUDEA = "bermudea"
    GATE_CD34TOTAL = "cd34total"
    GATE_CD34ALLPOS = "cd34allpos"
    GATE_CD34REAL = "cd34real"
    GATE_PROMY = "promy"
    GATE_GRANULOCYTES = "granulocytes"
    GATE_MONOCYTES = "monocytes"
    GATE_PROERY = "proery"
    GATE_CD34NORMAL = "cd34normal"
    PHONETYPE_GATES = [GATE_PROMY, GATE_GRANULOCYTES, GATE_MONOCYTES, GATE_PROERY, GATE_CD34NORMAL]
    PHONETYPE_GATES_WITH_BLASTS = PHONETYPE_GATES + [GATE_BLASTOTHER, GATE_BLAST34]
    BERMUDEA_GATES_WITH_BLASTS = [GATE_PROMY, GATE_GRANULOCYTES, GATE_MONOCYTES, GATE_PROERY, GATE_BLASTOTHER]
    CD34_GATES = [GATE_CD34NORMAL, GATE_BLAST34]
    PHONETYPE_SAMPLING_WEIGHTS = [0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8]
    PHONETYPE_SAMPLING_WEIGHTS_CD34 = [0.5, 0.8]
    PHONETYPE_SAMPLING_WEIGHTS_BLASTOTHER = [0.5, 0.8]

    GATE_RENAME_DICT = {
        "CD7pos": "CD7",
        "cd7pos": "CD7",
        "CD7 pos" : "CD7",
        "cd7": "CD7",
        "CD7+" : "CD7",
        "CD20 pos": "CD20",
        "blasten": "blast",
        "Blast other": "blastother",
        "Blast 34": "blast34",
        "Blasten": "blast",
        "blasts": "blast",
        "Blasts": "blast",
        "Blast": "blast",
        "BLasts" : "blast",
        "Syto41" : "Syto",
        "Syto pos": "Syto",
        "sytopos": "Syto",
        "Syto +": "Syto",
        "syto": "Syto",
        "Syto+": "Syto",
        "syto+": "Syto",
        "Syto +": "Syto",
        "CD19 pos": "CD19",
        "CD19+": "CD19",
        "CD19 +": "CD19",
        "cd19": "CD19",
        "cd19+": "CD19",
        "viable": "Intact",
        "intact": "Intact",
        "Viable": "Intact",
            "singlets": "Singlets",
        "mature B-cells": "mature B-Cells",
        "Plamsa cells": "Plasma Cells",
        "plasmacells": "Plasma Cells",
        "Plasmacells": "Plasma Cells",
        "P1_" : "P1",
        "Erythropoieses" : "Erythropoese",
        "Erythropoiese" : "Erythropoese"
    }

    def getBlastGates() -> List:
        return [GateCollection.GATE_BLAST34, 
                GateCollection.GATE_BLAST,
                GateCollection.GATE_BLASTEN,
                GateCollection.GATE_BLASTOTHER ]

    @staticmethod
    def renameGates(gates: pd.DataFrame):
        renameDict = GateCollection.GATE_RENAME_DICT
        renameDict = {k: v for k, v in renameDict.items() if k in gates.columns and not v in gates.columns}
        if len(renameDict) > 0:
            gates.rename(columns=renameDict, inplace=True)

        return gates
