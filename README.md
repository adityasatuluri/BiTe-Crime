# BIte CRIME: Bidirectional Interpretable Temporal Embedding Framework with GATv2 for Crime Prediction

## Requirements

To install requirements:

```setup
python3.8 -m venv venv
pip install -r requirements.txt
```

### Datasets are acquired from: [https://github.com/YeasirRayhanPrince/aist]

## Training

To train BiTe Crime on 2019 'theft' data of Near North Side, Chicago, run this command:

```train
python train.py
```

To further train BiTe Crime on different crime categories and communities of Chicago, run this command:

```train
python train.py --tct=chicago --tr=ID1 --tc=ID2
```

For IDs of the communities or the crime-categories, check `data/chicago/chicago_cid_to_name.txt` and `data/chicago/chicago_crime-cat_to_id.txt`
