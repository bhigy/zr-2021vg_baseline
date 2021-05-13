## Phonetic

To compute ABX score, we use the activations of the 2nd recurrent layer of the VG model.
To extract these activations for, e.g., the low-budget baseline, run:

```
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/phonetic/dev-clean \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/phonetic/dev-clean \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/phonetic/dev-other \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/phonetic/dev-other \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/phonetic/test-clean \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/phonetic/test-clean \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/phonetic/test-other \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/phonetic/test-other \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
```

## Lexical


## Syntactic

## Semantic

We compute sSIMI scores on the output of the attention layer of the VG model.
Such activations can be extracted, e.g. for the low-budget baseline, by running:

```
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/semantic/dev/synthetic \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/semantic/dev/synthetic \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/semantic/dev/librispeech \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/semantic/dev/librispeech \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/semantic/test/synthetic \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/semantic/test/synthetic \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/data/zerospeech2021/semantic/test/librispeech \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bertsmall-librispeech960/semantic/test/librispeech \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
```

## Full evaluation pipeline

```bash
# Checking format is OK
zerospeech2021-validate vg_net_15 --no-lexical --no-syntactic --only-dev

# Running the evaluation
zerospeech2021-evaluate /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021 submission_test/ --no-syntactic --no-lexical
```
