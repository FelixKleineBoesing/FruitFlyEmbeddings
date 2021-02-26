from pathlib import Path

import pandas as pd


def prepare_online_retail(path: Path = None):
    data = pd.read_excel(path, engine="openpyxl")
    data = data.loc[:, ["InvoiceNo", "StockCode"]]
    data["present"] = 1





if __name__ == "__main__":
    prepare_online_retail(Path("..", "data", "Online Retail.xlsx"))