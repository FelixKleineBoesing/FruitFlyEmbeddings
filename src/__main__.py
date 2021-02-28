from pathlib import Path

from cleaning import prepare_online_retail


def main():
    prepare_online_retail(Path("..", "data", "Online Retail.xlsx"),
                          Path("..", "data", "products.feather"))


if __name__ == "__main__":
    main()
