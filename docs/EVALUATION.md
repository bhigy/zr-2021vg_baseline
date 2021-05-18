In order to evaluate our systems, we need to create a folder following the submission format described on the [challenge's website](https://zerospeech.com/2021/instructions.html). We can then use the [command line tools](https://github.com/bootphon/zerospeech2021) provided with the challenge to validate/evaluate the submission.

## Generation of the submission files

### Phonetic evaluation

To compute ABX score, we use the activations of the 2nd recurrent layer of the VG model.
To extract these activations for, e.g., the low-budget baseline, run:

```
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/phonetic/dev-clean \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/phonetic/dev-clean \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/phonetic/dev-other \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/phonetic/dev-other \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/phonetic/test-clean \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/phonetic/test-clean \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/phonetic/test-other \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/phonetic/test-other \
    --batch_size 8 --layer rnn1 --output_file_extension '.txt' --file_extension '.wav'

# Cleaning to match submission format
cd data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/phonetic
mv dev-clean/rnn1/* dev-clean
mv dev-other/rnn1/* dev-other
mv test-clean/rnn1/* test-clean
mv test-other/rnn1/* test-other
rmdir */rnn1; rm */_info_args.json
cd ../../../..
```

### Lexical evaluation

Lexical evaluation is performed on pseudo-probabilities extracted from the language model.
To obtain them, we thus need to extract VG activation first and quantize them.
All this can be performed, e.g. for the low-budget baseline, by running:

```bash
# Extracting activations
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/lexical/dev \
    data/activations/vg-spokencoco/zerospeech2021/lexical/dev \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/lexical/test \
    data/activations/vg-spokencoco/zerospeech2021/lexical/test \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' --file_extension '.wav'

# Quantizing data
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/zerospeech2021/lexical/dev/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/lexical/dev \
    --recursionLevel 0
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/zerospeech2021/lexical/test/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/lexical/test \
    --recursionLevel 0

# Extracting LM pseudo-probabilities
python -m scripts.compute_proba_BERT \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/lexical/dev/quantized_outputs.txt \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/lexical/dev.txt \
    exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/checkpoint_best.pt \
    --dict  data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/dict.txt
python -m scripts.compute_proba_BERT \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/lexical/test/quantized_outputs.txt \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/lexical/test.txt \
    exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/checkpoint_best.pt \
    --dict  data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/dict.txt
```

### Syntactic evaluation

Syntactic evaluation is also performed on pseudo-probabilities extracted from the language model.
To perform necessary operations for, e.g., the low-budget baseline, run:

```bash
# Extracting activations
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/syntactic/dev \
    data/activations/vg-spokencoco/zerospeech2021/syntactic/dev \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/syntactic/test \
    data/activations/vg-spokencoco/zerospeech2021/syntactic/test \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' --file_extension '.wav'

# Quantizing data
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/zerospeech2021/syntactic/dev/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/syntactic/dev \
    --recursionLevel 0
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/zerospeech2021/syntactic/test/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/syntactic/test \
    --recursionLevel 0

# Extracting LM pseudo-probabilities
python -m scripts.compute_proba_BERT \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/syntactic/dev/quantized_outputs.txt \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/syntactic/dev.txt \
    exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/checkpoint_best.pt \
    --dict  data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/dict.txt
python -m scripts.compute_proba_BERT \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/zerospeech2021/syntactic/test/quantized_outputs.txt \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/syntactic/test.txt \
    exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/checkpoint_best.pt \
    --dict  data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/dict.txt
```

### Semantic evaluation

We compute sSIMI scores on the output of the attention layer of the VG model.
Such activations can be extracted, e.g. for the low-budget baseline, by running:

```
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/semantic/dev/synthetic \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/semantic/dev/synthetic \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/semantic/dev/librispeech \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/semantic/dev/librispeech \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/semantic/test/synthetic \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/semantic/test/synthetic \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/zerospeech2021/semantic/test/librispeech \
    data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/semantic/test/librispeech \
    --batch_size 8 --layer att --output_file_extension '.txt' --file_extension '.wav'

# Cleaning to match submission format
cd data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960/semantic
mv dev/librispeech/att/* dev/librispeech;
mv dev/synthetic/att/* dev/synthetic;
mv test/librispeech/att/* test/librispeech;
mv test/synthetic/att/* test/synthetic;
rmdir */*/att; rm */*/_info_args.json
cd ../../../..
```

### Additional files

We finally need to create the metadata file `meta.yaml` and the file `code/README` with a link to the code.
To do this, run following code:

```bash
cd data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960
cat > meta.yaml <<'EOF'
author: Zero Speech Challenge Organizers
affiliation: Aalto University, CNRS, EHESS, ENS, Inria, PSL Research University, Tampere University, Tilburg University, University of Texas
description: Low-budget baseline
open_source: True
train_set: SpokenCOCO, Librispeech
gpu_budget: 72.0
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.02
  semantic:
    metric: cosine
    pooling: max
EOF
mkdir code
echo "https://github.com/bhigy/zr-2021vg_baseline" > code/README
cd ../../..
```

## Validation/evaluation

The submission folder can be validated and evaluated with following commands:

```bash
mkdir results/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960
cd results/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960
zerospeech2021-validate ~/corpora/zerospeech2021 data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960
zerospeech2021-evaluate ~/corpora/zerospeech2021 data/submission/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960
cd ../..
```
