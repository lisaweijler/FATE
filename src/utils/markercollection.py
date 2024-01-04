from typing import List
import pandas as pd


class MarkerCollection:
    """
    MarkerCollection holds translations from different occuring marker names to the standarised names.
    """

    MARKER_DICT = {
        "13": "CD13",

        "CD1A": "CD1A",
        "CD1AAPC": "CD1A",
        "CD1AAPC-A": "CD1A",
        "CD1ABV605-A": "CD1A",
        "CD1AAPC-H": "CD1A-H",
        "CD1ABV605-H": "CD1A-H",

        "CD3BV605": "CD3",
        "CD3PG": "CD3",
        "CD3PC5.5-A": "CD3",
        "CD3BV605VIOLET610-A": "CD3",
        "CD3PERCP-CY5.5-A": "CD3",
        "CD3BV605VIOLET610-H": "CD3-H",
        "CD3PERCP-CY5.5-H": "CD3-H",
        "CD3PC5.5-H": "CD3-H",

        "CD4AX700": "CD4",
        "CD4APC-A700-A": "CD4",
        "CD4AX700APC-A700-A": "CD4",
        "CD4BV711-A": "CD4",
        "CD4V450-A": "CD4",
        "CD4BV711-H": "CD4-H",
        "CD4V450-H": "CD4-H",
        "CD4AX700APC-A700-H": "CD4-H",

        "CD5PE": "CD5",
        "CD5PC7": "CD5",
        "CD5 PC7-A": "CD5",
        "CD5ECD": "CD5",
        "CD5PC7-A": "CD5",
        "CD5PE-CY7-A": "CD5",
        "CD5PC7-H": "CD5-H",
        "CD5PE-CY7-H": "CD5-H",

        "CD7ECD": "CD7",
        "CD7PC55": "CD7",
        "CD7APC-A": "CD7",
        "CD7 APC-A" : "CD7",
        "CD7APCAPC-A": "CD7",
        "CD7PEDAZZLE594ECD-A": "CD7",
        "CD7PE-A": "CD7",
        "CD7PE-H": "CD7-H",
        "CD7APCAPC-H": "CD7-H",
        "CD7PEDAZZLE594ECD-H": "CD7-H",
        "CD7APC-H": "CD7-H",

        "CD7-CD13": "CD7CD13",

        "CD7_19": "CD7CD19",
        "CD7_CD19": "CD7CD19",
        "CD19+7-PE": "CD7CD19",
        "CD7+19": "CD7CD19",
        "CD19+CD7-PE": "CD7CD19",
        "CD7+CD19": "CD7CD19",

        "CD7_56": "CD7CD56",
        "CD7/56": "CD7CD56",
        "CD7VCD56": "CD7CD56",
        "CD56CD7": "CD7CD56",
        "CD7/56APC-A": "CD7CD56",
        "CD7/56APC-H": "CD7CD56-H",
        "CD7/CD56APC-H": "CD7CD56-H",
        "CD7/CD56APC-A": "CD7CD56",

        "CD8AX750": "CD8",
        "CD8APC-CY7APC-A750": "CD8",
        "CD8APC-CY7APC-A750-A": "CD8",
        "CD8APC-H7-A": "CD8",
        "CD8APC-H7-H": "CD8-H",

        "CD10PE": "CD10",
        "CD10PC5": "CD10",
        "CD10PC55": "CD10",
        "CD10VIOLET610-A": "CD10",
        "CD101UL": "CD10",
        "CD10VIOLET610-H": "CD10-H",
        "CD10BV605VIOLET610-H": "CD10-H",
        "CD10BV605VIOLET610-A": "CD10",

        "CD10+16": "CD10CD16",

        "CD10+CD371": "CD10CD371",

        "CD11A": "CD11A",
        "CD11a": "CD11A",
        "CD11APE": "CD11A",
        "CD11APE-A": "CD11A",
        "CD11APE-H": "CD11A-H",

        "CD11b": "CD11B",
        "CD11B-APC-A750": "CD11B",
        "CD11B-A750": "CD11B",
        "CD11BAPC-A750-A": "CD11B",
        "CD11BAPC-A750-H": "CD11B-H",

        "CD11C": "CD11C",

        "CD13-APC": "CD13",
        "CD13APC-A": "CD13",
        "CD13APC-H": "CD13-H",

        "CD13_19": "CD13CD19",

        "CD14-APC-A700": "CD14",
        "CD14APC-A700-A": "CD14",
        "CD14-A700": "CD14",
        "CD14APC-A700-H": "CD14-H",

        "CD15-FITC": "CD15",
        "CD15FITC-A": "CD15",
        "CD15FITC-H": "CD15-H",

        "CD16BV605": "CD16",
        "CD16VIOLET610-A": "CD16",
        "CD16VIOLET610-H": "CD16-H",

        "CD19PECY7": "CD19",
        "CD19PC7": "CD19",

        "CD19/56": "CD19CD56",
        "CD19+CD56PE-A": "CD19CD56",
        "CD19+CD56PE-H": "CD19CD56-H",

        "CD20 LOG": "CD20",
        "CD20APC-A750": "CD20",
        "CD20AX750": "CD20",
        "CD20BC": "CD20",

        "CD24VIOLET780-A": "CD24",
        "CD24BV785VIOLET780-A": "CD24",
        "CD24VIOLET780-H": "CD24-H",
        "CD24BV785VIOLET780-H": "CD24-H",

        "CD33-PC7": "CD33",
        "CD33PE": "CD33",
        "CD33VIOLET660-A": "CD33",
        "CD33BV650VIOLET660-A": "CD33",
        "CD33 PC7-A": "CD33",
        "CD33PC7-A": "CD33",
        "CD33PC7-H": "CD33-H",
        "CD33 PC7-H": "CD33-H",
        "CD33VIOLET660-H": "CD33-H",
        "CD33BV650VIOLET660-H": "CD33-H",

        "CD34-ECD": "CD34",
        "CD34ECDECD-A": "CD34",
        "CD34ECD": "CD34",
        "CD34BV605": "CD34",
        "CD34ECD-A": "CD34",
        "CD34 ECD-A": "CD34",
        "CD34ECD-H": "CD34-H",
        "CD34 ECD-H": "CD34-H",
        "CD34ECDECD-H": "CD34-H",

        "CD34+CD99": "CD34CD99",

        "CD38 LOG": "CD38",
        "CD38-FITC": "CD38",
        "CD38APC-A700": "CD38",
        "CD38BC1UL": "CD38",
        "CD38APC-AX700": "CD38",
        "CD38AX700": "CD38",
        "CD38 FITC-A": "CD38",
        "CD38FITC-A": "CD38",
        "CD381UL": "CD38",
        "CD38FITC-H": "CD38-H",
        "CD38 FITC-H": "CD38-H",

        "CD45 LOG": "CD45",
        "CD45-KRO": "CD45",
        "CD45KA": "CD45",
        "CD45KO525-A": "CD45",
        "CD45PERCPPC5.5-A": "CD45",
        "CD45 KO525-A": "CD45",
        "CD45V500-C-A": "CD45",
        "CD45 KO525-H": "CD45-H",
        "CD45KO525-H": "CD45-H",
        "CD45PERCPPC5.5-H": "CD45-H",
        "CD45V500-C-H": "CD45-H",

        "CD45RA-A750": "CD45RA",
        "CD45RA-APC-A750": "CD45RA",
        "CD45RABV510": "CD45RA",
        "CD45RA APC-A750-A": "CD45RA",
        "CD45RAAPC-A750-A": "CD45RA",
        "CD45RAAPC-A750-H": "CD45RA-H",
        "CD45RA APC-A750-H": "CD45RA-H",

        "CD48BL": "CD48",
        "CD48PE-A": "CD48",
        "CD48 PE-A": "CD48",
        "CD48FITC-A": "CD48",
        "CD48APC-A": "CD48",
        "CD48APC-H": "CD48",
        "CD48PE-H": "CD48-H",

        "CD56APC-A": "CD56",
        "CD56PE-A": "CD56",
        "CD56PE-H": "CD56-H",
        "CD56APC-H": "CD56-H",

        "CD16+56": "CD56CD16",
        "CD56CD16": "CD56CD16",
        "CD56CD16KO": "CD56CD16",
        "CD56+16": "CD56CD16",
        "CD56+CD16": "CD56CD16",
        "CD16+CD56APC": "CD56CD16",
        "CD16CD56": "CD56CD16",
        "CD56+16BV510KO525-A": "CD56CD16",
        "CD16/56BV605VIOLET610-A": "CD56CD16",
        "CD16/56VIOLET610-A": "CD56CD16",
        "CD16+56APC-R700-A": "CD56CD16",
        "CD16+CD56PE-A": "CD56CD16",
        "CD56+16BV510KO525-H": "CD56CD16-H",
        "CD16/56BV605VIOLET610-H": "CD56CD16-H",
        "CD16+CD56PE-H": "CD56CD16-H",
        "CD16/56VIOLET610-H": "CD56CD16-H",
        "CD16+56APC-R700-H": "CD56CD16-H",

        "CD58FITC": "CD58",

        "CD64PE-A": "CD64",
        "CD64PE-H": "CD64-H",

        "CD71ULBIOLEGEND": "CD71",
        "CD71VIOLET660-A": "CD71",
        "CD71VIOLET660-H": "CD71-H",

        "CD79A": "CD79A",

        "CD96PE-A": "CD96",
        "CD96PE-H": "CD96-H",

        "CD99APC": "CD99",
        "CD99PE": "CD99",
        "CD99 PE" : "CD99",
        "CD99PE-A": "CD99",
        "CD99FITC": "CD99",
        "CD99-PE": "CD99",
        "CD99 FITC-A": "CD99",
        "CD99FITC-A": "CD99",
        "CD99 APC-A": "CD99",
        "CD99APC-A": "CD99",
        "CD992UL": "CD99",
        "CD99BC": "CD99",
        "CD99APC-H": "CD99-H",
        "CD99 APC-H": "CD99-H",
        "CD99PE-H": "CD99-H",
        "CD99FITC-H": "CD99-H",

        "CD117-PC5.5": "CD117",
        "CD117PC5.5-A": "CD117",
        "CD117BV605": "CD117",
        "CD117 PC5.5-A": "CD117",
        "CD117 PC5.5-H": "CD117-H",
        "CD117PC5.5-H": "CD117-H",

        "CD123-APC-A700": "CD123",
        "CD123-A700": "CD123",
        "CD123PE": "CD123",
        "CD123 APC-A700-A": "CD123",
        "CD123APC-A700-A": "CD123",
        "CD123APC-A700-H": "CD123-H",
        "CD123 APC-A700-H": "CD123-H",
        "CD123PE-H": "CD123-H",
        "CD123PE-A": "CD123",

        "CD133VIOLET780-A": "CD133",
        "CD133VIOLET780-H": "CD133-H",

        "CD312APC": "CD312",
        "CD312APC-A": "CD312",
        "CD312APC-H": "CD312-H",

        "CD371-APC": "CD371",
        "CLL1-APC": "CD371",
        "CLL1": "CD371",
        "CLL-1": "CD371",
        "CLL": "CD371",
        "CD371 PE-A": "CD371",
        "CD371PE-A": "CD371",
        "CD371PE-H": "CD371-H",
        "CD371 PE-H": "CD371-H",
        "CD37371": "CD371",

        "FSINT": "FSC-A",
        "FS INT": "FSC-A",
        "FSPEAK": "FSC-H",
        "FS PEAK": "FSC-H",
        "FSTOF": "FSC-W",
        "FS TOF": "FSC-W",
        "FSC-WIDTH": "FSC-W",
        "FSC-Width": "FSC-W",

        "HLA-DR-PB": "HLA-DR",
        "HLADR-PB": "HLA-DR",
        "HLADR": "HLA-DR",
        "HLA-DR PB450-A": "HLA-DR",
        "HLA-DRPB450-A": "HLA-DR",
        "HLA-DR PB450-H": "HLA-DR-H",
        "HLA-DRPB450-H": "HLA-DR-H",
        "HLADRPB450-A": "HLA-DR",
        "HLADRPB450-H": "HLA-DR-H",

        "ICD3": "ICD3",

        "NG2PE-A": "NG2",
        "NG2PE-H": "NG2-H",

        "SSINT": "SSC-A",
        "SS INT": "SSC-A",
        "SSPEAK": "SSC-H",
        "SS PEAK": "SSC-H",
        "SSTOF": "SSC-W",
        "SS TOF": "SSC-W",

        "SYTO 41": "SY41",
        "Syto 41": "SY41",
        "_SYTO41-A": "SY41",
        "_Syto41-A" : "SY41",
        "SY41 LOG": "SY41",
        "SYTO41 LOG": "SY41",
        "Syto41 PB450-A": "SY41",
        "SYTO41PB450-A": "SY41",
        "SYTO41V450-A": "SY41",
        "SYTO41PB450-H": "SY41-H",
        "SYTO41V450-H": "SY41-H"
    }

    @staticmethod
    def renameMarkers(events: pd.DataFrame):

        renameDict = {k: v for k, v in MarkerCollection.MARKER_DICT.items() if k in events.columns and not v in events.columns}
        if len(renameDict) > 0:
            events.rename(columns=renameDict, inplace=True)

        return events
    
    @staticmethod
    def renameMarkers_after_preload(markerlist: List[str]):

        renameDict = {k: v for k, v in MarkerCollection.MARKER_DICT.items() if k in markerlist and not v in markerlist}
        if len(renameDict) > 0:
            renamed_markerlist = [renameDict[k] if k in renameDict else k for k in markerlist]

            return renamed_markerlist
        return markerlist
