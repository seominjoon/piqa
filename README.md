### Download requirements
Make sure you have python 3.x.
```bash
chmod +x download.sh; ./download.sh
```

### Train
Let `$SQUAD_DIR` be the directory that has both train and dev json files of SQuAD.

For LSTM model:
```bash
python main.py --cuda --data_dir $SQUAD_DIR
```

For LSTM+SA model:
```bash
python main.py --cuda --num_heads 2 --data_dir $SQUAD_DIR
```

For LSTM+SA+ELMo model:
```bash
python main.py --cuda --num_heads 2 --elmo --data_dir $SQUAD_DIR
```

