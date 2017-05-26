### Introduction
This project is on-going project. Currently it goes on table features computation. We use DataFrame (supported by [Pandas](http://pandas.pydata.org)) as the main dataType for our data.

### Requirements
1. pandas >= 0.20.1

### Usage
use data_profile.py to profile the csv file into json format. Command line usage is as following:

```sh
python data_profile.py data.csv profiled_data.json
```

### Format
the output JSON format:

1. for columns


```json
{
  "column_id": {
    "num_missing": "the number of missing values in this column",
    "language": "language code, en, sp, etc.",
    "length": {
      "character": {
        "average": "average number of chars in every cell, in this column",
        "standard-deviation": "standard deviation of the number of chars in cells, in this column"
      },
      "token": {
        "average": "average number of tokens, separating by blank or punctuation",
        "standard-deviation": "token standard deviation"
      }
    },
    "num_integer": "the number of cell that it contains integer",
    "num_decimal": "the number of cell that it contains decimal",
    "num_distinct_values": "the number of distinct values (consider the content in a cell as a value)",
    "num_distinct_tokens": "same as num_distinct_values, but consider each token as a value",
    "frequent-entries": {
      "most_common_values": {
        "value-1": "count 1",
        "value-2": "count-2",
        "value-k": "count-4"
      },
      "most_common_tokens": {
        "token-1": "count 1",
        "token-2": "count-2",
        "token-k": "count-4"
      },
      "most_common_punctuation": {
        "token-1": "count 1",
        "token-2": "count-2",
        "token-k": "count-4"
      },
      "most_common_alphanumeric_tokens": {
        "token-1": "count 1",
        "token-2": "count-2",
        "token-k": "count-4"
      },
      "most_common_numeric_tokens": {
        "token-1": "count 1",
        "token-2": "count-2",
        "token-k": "count-4"
      }
    }
  }
}
```

notes:

1. token: delimiter is blank space