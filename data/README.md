# Evaluation datasets

## Olist Brazilian E-Commerce

Place the official Kaggle CSVs in this directory (`./data/`).

### Download

1. Go to [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Download and unzip into `./data/` so the following files exist:

```
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

Or with the Kaggle CLI (requires `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data --unzip
```

Runtime code assumes these files are already present (no network access at evaluation time).

## DataCo Smart Supply Chain

Second-dataset replication (§5.9). Place
`DataCoSupplyChainDataset.csv` under [`dataco/`](dataco/) — see
[`dataco/README.md`](dataco/README.md).
