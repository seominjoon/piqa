cl add bundle squad-data//dev-v1.1.json .
cl run piqa-master:piqa-master dev-v1.1.json:dev-v1.1.json  "python piqa-master/squad/split.py dev-v1.1.json dev-v1.1-context.json dev-v1.1-question.json" -n run-split
cl make run-split/dev-v1.1-context.json -n dev-v1.1-context.json
cl make run-split/dev-v1.1-question.json -n dev-v1.1-question.json

# For LSTM Model
# cl run dev-v1.1-context.json:dev-v1.1-context.json piqa-master:piqa-master model.pt:model.pt "python piqa-master/squad/main.py baseline --cuda --static_dir /static --load_dir model.pt --mode embed_context --test_path dev-v1.1-context.json --context_emb_dir context" -n run-context --request-docker-image minjoon/research:180908 --request-memory 8g --request-gpus 1
# cl run dev-v1.1-question.json:dev-v1.1-question.json piqa-master:piqa-master model.pt:model.pt "python piqa-master/squad/main.py baseline --cuda --batch_size 256 --static_dir /static --load_dir model.pt --mode embed_question --test_path dev-v1.1-question.json --question_emb_dir question" -n run-question --request-docker-image minjoon/research:180908 --request-memory 8g --request-gpus 1

# For LSTM+SA+ELMo Model
cl run dev-v1.1-context.json:dev-v1.1-context.json piqa-master:piqa-master model.pt:model.pt "python piqa-master/squad/main.py baseline --elmo --num_heads 2 --batch_size 32 --cuda --static_dir /static --load_dir model.pt --mode embed_context --test_path dev-v1.1-context.json --context_emb_dir context" -n run-context --request-docker-image minjoon/research:180908 --request-disk 4g --request-memory 8g --request-gpus 1
cl run dev-v1.1-question.json:dev-v1.1-question.json piqa-master:piqa-master model.pt:model.pt "python piqa-master/squad/main.py baseline --elmo --num_heads 2 --batch_size 128 --cuda --static_dir /static --load_dir model.pt --mode embed_question --test_path dev-v1.1-question.json --question_emb_dir question" -n run-question --request-docker-image minjoon/research:180908 --request-disk 4g --request-memory 8g --request-gpus 1

cl run piqa-master:piqa-master dev-v1.1.json:dev-v1.1.json run-context:run-context run-question:run-question "python piqa-master/squad/merge.py dev-v1.1.json run-context/context run-question/question pred.json" -n run-merge --request-docker-image minjoon/research:180908 --request-disk 4g
cl make run-merge/pred.json -n predictions-ELMo

cl macro squad-utils/dev-evaluate-v1.1 predictions-ELMo
cl edit predictions-ELMo --tags squad-test-submit pi-squad-test-submit