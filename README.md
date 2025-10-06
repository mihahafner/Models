# Models
"""
+------------------------------------------------+
|                Your Workspace                  |
|------------------------------------------------|
| Models/                                        |
|  ├── Transformer.py   ← (Core architecture)    |
|  ├── train_timeseries.py ← (Training script)   |
|  └── __pycache__/                             |
|                                                |
| Data/                                          |
|  └── (optional) dataset files                  |
+------------------------------------------------+

Your system currently includes:

    Core Model (Transformer.py) — Implements the Transformer encoder-decoder architecture for general-purpose sequence modeling.

    Wrapper (Seq2SeqForecaster) — A higher-level adapter around the Transformer that makes it easy to apply to time series or any numeric sequence task.

    Train Script (train_timeseries.py) — Generates synthetic sine-wave data, trains the model, and evaluates it.




"""