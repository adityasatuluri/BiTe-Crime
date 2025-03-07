# BIte CRIME: Bidirectional Interpretable Temporal Embedding Framework with GATv2 for Crime Prediction

This repository contains the code for [BIte CRIME: Bidirectional Interpretable Temporal Embedding Framework with GATv2 for Crime Prediction](https://google.com/).

## Requirements

To install requirements:

```setup
python3.8 -m venv venv
pip install -r requirements.txt
```

## Training

To train AIST on 2019 'theft' data of Near North Side, Chicago, run this command:

```train
python train.py
```

To further train AIST on different crime categories and communities of Chicago, run this command:

```train
python train.py --tct=chicago --tr=ID1 --tc=ID2
```

For IDs of the communities or the crime-categories, check `data/chicago/chicago_cid_to_name.txt` and `data/chicago/chicago_crime-cat_to_id.txt`
