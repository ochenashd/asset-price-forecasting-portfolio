from src.database.connection import DatabaseConfig, create_database_engine, test_connection
from src.database.repository import AssetRepository


def main() -> None:
    config = DatabaseConfig.from_env()
    engine = create_database_engine(config)

    if not test_connection(engine):
        raise RuntimeError("Database connection failed.")

    asset_repo = AssetRepository(engine)

    asset_id = asset_repo.upsert_asset(
        symbol="BTCUSDT",
        asset_name="Bitcoin",
        asset_type="crypto",
        exchange="Binance",
        quote_currency="USDT",
    )

    asset = asset_repo.get_asset_by_symbol("BTCUSDT")

    print(f"Asset ID: {asset_id}")
    print(asset)


if __name__ == "__main__":
    main()