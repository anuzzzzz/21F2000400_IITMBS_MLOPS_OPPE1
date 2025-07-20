from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64, ValueType
from datetime import timedelta

# Define entity
stock_entity = Entity(name="stock", value_type=ValueType.STRING)

# Define feature sources
stock_source_v0 = FileSource(
    name="stock_source_v0",
    path="../data/processed/processed_v0.csv",
    timestamp_field="timestamp",
)

stock_source_v1 = FileSource(
    name="stock_source_v1", 
    path="../data/processed/processed_v1.csv",
    timestamp_field="timestamp",
)

# Define feature views
stock_features_v0 = FeatureView(
    name="stock_features_v0",
    entities=[stock_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
        Field(name="target", dtype=Int64),
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64),
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
    ],
    online=True,
    source=stock_source_v0,
)

stock_features_v1 = FeatureView(
    name="stock_features_v1",
    entities=[stock_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
        Field(name="target", dtype=Int64),
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64),
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
    ],
    online=True,
    source=stock_source_v1,
)
