from src.data_preprocessing import load_data


def test_load():
    df = load_data("data/raw/used_car_price_dataset_extended.csv")
    assert df is not None