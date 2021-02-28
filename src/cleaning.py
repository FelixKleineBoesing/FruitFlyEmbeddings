from pathlib import Path

import pandas as pd


def prepare_online_retail(path: Path, output_path: Path):
    data = pd.read_excel(path, engine="openpyxl")
    data = data.loc[:, ["InvoiceNo", "StockCode"]]
    data["present"] = 1
    number_products = data.groupby("StockCode").agg(Count=("present", sum))
    products_sold_more_than_once = number_products[number_products["Count"] > 1].index.tolist()
    data = data.loc[data["StockCode"].isin(products_sold_more_than_once), ["StockCode", "InvoiceNo"]]
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data["StockCode"] = data["StockCode"].astype(str)
    data["StockCode"] = data["StockCode"].str.replace(" ", "")
    concatenated_data = data.groupby(['InvoiceNo'])['StockCode'].agg(' '.join).reset_index()
    concatenated_data.to_feather(output_path)


if __name__ == "__main__":
    prepare_online_retail(Path("..", "data", "Online Retail.xlsx"),
                          Path("..", "data", "products.feather"))