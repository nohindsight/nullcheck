# Contributing

Thanks for helping. Reports of **false alarms** and **missed leaks** are the most useful contributions.

## Reporting a bug

Open an issue with:
- a small **synthetic** example: fictional or generated prices (a few hundred rows is plenty) and the shortest feature function or strategy that shows the problem;
- the exact command or Python call, and the full report text;
- `python -m pip show nullcheck` version, and your Python, pandas and numpy versions (plus backtesting if used).

Never attach the private `--out` folder, your real data or your real strategy: the folder contains a copy of your data rows and the paths of your code files.

## Changing code

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[backtesting]"
python -m unittest discover -v
```

- Keep changes small and add a regression test that fails before your change.
- Do not paste code from backtesting.py or any other copyleft project into this repository; importing an installed copy is fine.
- Contributions are accepted under the MIT licence of this repository.

Scope: nullcheck checks prefix invariance of values computed from raw rows, and for backtesting.py only up to the end of `Strategy.init()`. Checking `next()`, orders or trade outcomes is out of scope for now.
