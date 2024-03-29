Dependencies can be installed using following commands:

```bash
conda create --name zrvg python=3.8 && conda activate zrvg
xargs -L 1 pip install < requirements.txt  # Forces to install packages in order
```

This step will install 4 main repositories :
* [platalea](https://github.com/spokenlanguage/platalea/tree/zerospeech21-vg) to train train visually grounded models.
* [fairseq](https://github.com/fairseq/fairseq) to train language models.
* [CPC_audio](https://github.com/tuanh208/CPC_audio/tree/zerospeech) to use pretrained CPC and discretize representations.
* [zerospeech 2021](https://github.com/bootphon/zerospeech2021) to evaluate models.

You will also need to add this repository to your `PYTHONPATH`.
