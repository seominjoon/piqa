### Download requirements
Make sure you have python 3.x.
```bash
chmod +x download.sh; ./download.sh
```

### Train
For LSTM model:
```bash
python main.py --cuda
```

For LSTM+SA model:
```bash
python main.py --cuda --num_heads 2
```

For LSTM+SA+ELMo model:
```bash
python main.py --cuda --num_heads 2 --elmo
```

