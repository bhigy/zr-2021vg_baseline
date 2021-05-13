## Full evaluation pipeline

```
# Extract activations for ABX task
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/phonetic/dev-clean vg_net_15/phonetic/dev-clean --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/phonetic/dev-other vg_net_15/phonetic/dev-other --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav

# Extract activations for sSIMI task
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/semantic/dev/librispeech vg_net_15/semantic/dev/librispeech --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/semantic/dev/synthetic vg_net_15/semantic/dev/synthetic --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav

# Checking format is OK
zerospeech2021-validate vg_net_15 --no-lexical --no-syntactic --only-dev

# Running the evaluation
zerospeech2021-evaluate /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021 submission_test/ --no-syntactic --no-lexical
```
