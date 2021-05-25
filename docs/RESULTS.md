Each of the 4 considered metrics are shallowly described at the beginning of each section. Please note that some approximations are made in favour of clarity.
Please refer to the [Zero Resource Speech paper](https://arxiv.org/pdf/2011.11588.pdf) for more details.

Checkpoints of the baseline can be obtained by running :

```bash
curl -L -o exps.zip https://www.dropbox.com/sh/xcdtml8a3go3fk8/AAAfymU80lS6ZKMr3y9okwApa?dl=0
unzip exps.zip -d exps
```



# ABX error rate : Phoneme discriminability

In this task, the model receives 3 triplets A, B and X that are triphones, with A and X being the same triphone (but different occurences), and B differing from A/X only in its central phone (e.g /big/ vs /bug/).
Under a distance function d, we expect d(A,X) < d(B,X) as A and X correspoond to the same triphone. This metric is computed over 2 conditions : **within speakers**, when the 3 triphones are pronounced by the same speakers, and **across speakers** when triphones are pronounced by different speakers. Lower is better.

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="4" align="center" style="font-weight:bold">LibriSpeech dev</td>
  </tr>

  <tr>
    <td></td>
    <td></td>
    <td colspan="2" align="center" style="font-weight:bold">dev-clean</td>
    <td colspan="2" align="center" style="font-weight:bold">dev-other</td>
  </tr>

  <tr>
    <td style="font-weight:bold">Model</td>
    <td style="font-weight:bold">Layer</td>
    <td style="font-weight:bold">within</td>
    <td style="font-weight:bold">across</td>
    <td style="font-weight:bold">within</td>
    <td style="font-weight:bold">across</td>
  </tr>

  <tr>
    <td>MFCCs+VG</td>
    <td>rnn1</td>
    <td>8.70</td>
    <td>10.69</td>
    <td>9.86</td>
    <td>14.71</td>
  </tr>

  <tr>
    <td>CPC small+VG</td>
    <td>rnn1</td>
    <td>5.36</td>
    <td>6.68</td>
    <td>7.41</td>
    <td>11.30</td>
  </tr>
</table>

**Layer** refers to the layer used to extract representations ('rnn1' corresponds to the second recurrent layer).

# sSIMI : Semantic similarity with human judgments

In this task, the model receives 2 words, let's say /abduct/ and /kidnap/.
The distance between the embeddings of these 2 words are computed.
Then, a Spearman's rank correlation coefficent is computed between these distances, and human semantic similarity judgements.
Higher is better.

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="2" align="center" style="font-weight:bold">sSIMI dev</td>
  </tr>

  <tr>
    <td style="font-weight:bold">Model</td>
    <td style="font-weight:bold">Layer</td>
    <td align="center" style="font-weight:bold">librispeech</td>
    <td align="center" style="font-weight:bold">synthetic</td>
  </tr>

  <tr>
    <td>MFCCs+VG</td>
    <td>att</td>
    <td>13.0894</td>
    <td>9.4661</td>
  </tr>

  <tr>
    <td>CPC small+VG</td>
    <td>att</td>
    <td>11.8885</td>
    <td>6.3074</td>
  </tr>
</table>

**Layer** refers to the layer used to extract representations ('att' corresponds to the attention layer).


# sBLIMP : Syntax acceptability judgment

In this task, the model receives two sentences, one of which is syntactically wrong. Let's say /dogs eat meat/ vs /dogs eats meat/.
The model is asked to return pseudo-probabilities for each of these sentence. The pseudo-probability of the syntactically right sentence is expected to be higher than the pseudo-probability of the syntactically wrong sentence.
Note that the hyperparameters used to extract the pseudo-probabilities from the model can greatly impact the performance.
Higher is better.

<table>
  <tr>
    <td style="font-weight:bold">Model</td>
    <td style="font-weight:bold">K</td>
    <td style="font-weight:bold">M_d</td>
    <td style="font-weight:bold">Delta_t</td>
    <td colspan="1" align="center" style="font-weight:bold">sBLIMP dev</td>
  </tr>


  <tr>
    <td>MFCCs+VG+<br>KMEANS+<br>BERT small</td>
    <td>50</td>
    <td>10</td>
    <td>1</td>
    <td>53.19</td>
  </tr>

  <tr>
    <td>CPC small+VG+<br>KMEANS+<br>BERT large</td>
    <td>500</td>
    <td>10</td>
    <td>1</td>
    <td>54.68</td>
  </tr>
</table>

**K** refers to the number of clusters used in K-means.
**M_d** and **Delta_t** are respectively the decoding span size and the temporal sliding size used to extract pseudo-probabilities

# sWUGGY : Spot-the-word task

In this task, the model receives a word and a non-word. Let's say /brick/ and /blick/.
It it asked to return pseudo-probabilities for each of these. The pseudo-probability of the word is expected to be higher
than the pseudo-probability of the non-word.
Higher is better.


<table>
  <tr>
    <td style="font-weight:bold">Model</td>
    <td style="font-weight:bold">K</td>
    <td style="font-weight:bold">M_d</td>
    <td style="font-weight:bold">Delta_t</td>
    <td colspan="1" align="center" style="font-weight:bold">sWUGGY dev</td>
  </tr>


  <tr>
    <td>MFCCs+VG+<br>KMEANS+<br>BERT small</td>
    <td>50</td>
    <td>10</td>
    <td>1</td>
    <td>52.53</td>
  </tr>

  <tr>
    <td>CPC small+VG+<br>KMEANS+<br>BERT large</td>
    <td>500</td>
    <td>10</td>
    <td>1</td>
    <td>67.16</td>
  </tr>
</table>

**K** refers to the number of clusters used in K-means.
**M_d** and **Delta_t** are respectively the decoding span size and the temporal sliding size used to extract pseudo-probabilities

# Test metrics

<table>
<tr>
    <td style="font-weight:bold"></td>
    <td colspan="2" align="center" style="font-weight:bold">ABX test-clean</td>
    <td colspan="2" align="center" style="font-weight:bold">ABX test-clean</td>
    <td colspan="2" align="center" style="font-weight:bold">sSIMI</td>
    <td colspan="1" align="center" style="font-weight:bold"></td>
    <td colspan="1" align="center" style="font-weight:bold"></td>
  </tr>
  <tr>
    <td style="font-weight:bold">Model</td>
    <td colspan="1" align="center" style="font-weight:bold">within</td>
    <td colspan="1" align="center" style="font-weight:bold">across</td>
    <td colspan="1" align="center" style="font-weight:bold">within</td>
    <td colspan="1" align="center" style="font-weight:bold">across</td>
    <td colspan="1" align="center" style="font-weight:bold">librispeech</td>
    <td colspan="1" align="center" style="font-weight:bold">synthetic</td>
    <td colspan="1" align="center" style="font-weight:bold">sWUGGY</td>
    <td colspan="1" align="center" style="font-weight:bold">sBLIMP</td>
  </tr>


  <tr>
    <td>MFCCs+VG+<br>KMEANS+<br>BERT small</td>
    <td>8.39</td>
    <td>10.59</td>
    <td>10.66</td>
    <td>15.03</td>
    <td>-0.10</td>
    <td>9.99</td>
    <td>52.86</td>
    <td>53.02</td>
  </tr>

  <tr>
    <td>CPC small+VG+<br>KMEANS+<br>BERT large</td>
    <td>5.36</td>
    <td>6.71</td>
    <td>7.35</td>
    <td>11.92</td>
    <td>0.16</td>
    <td>9.71</td>
    <td>67.20</td>
    <td>54.53</td>
  </tr>
</table>


# Findings

Overall, using the visual modality for learning speech representations in an unsupervised way seems on par with audio only models.

* The ABX error rate obtained by CPC small is further improved with the VG model (high-budget baseline): it goes from 6.24% ABX error rate to 5.36% ABX error rate (librispeech dev-clean, within speakers).

* The best achievement is obtained on the semantic similarity task for which our best VG model gets a score of 13.09 compared to 8.72 for the audio-only baseline (results reported here are computed on sSIMI librispeech dev set).

* No improvement is observed on the syntax acceptability judgment task (sBLIMP). A small decrease is observed on the spot-the-word task : 70.69% for the audio-only baseline compared to 67.16% for the multimodal baseline.
