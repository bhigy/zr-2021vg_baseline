Each of the 4 considered metrics are shallowly described at the beginning of each section. Please note that some approximations are made in favour of clarity.
Please refer to the [Zero Resource Speech paper]((More details in the [ZR paper](https://arxiv.org/pdf/2011.11588.pdf))) for more details.


# ABX error rate : Phoneme discriminability

In this task, the model receives 3 triplets A, B and X that are triphones. 
With A and X being the same triphone (but different occurence), and B differing only in its center phone with A (/big/ vs /bug/).
Under a distance function d, we expect d(A,X) < d(B,X) as A and X are the same triphones. This metric is computed over 2 conditions : **within speakers**, when the 3 triphones are pronounced by the same speakers, and **across speakers** when triphones are pronounced by different speakers. 
Lower is better.

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="4" align="center">LibriSpeech dev</td>
  </tr>

  <tr>
    <td></td>
    <td></td>
    <td colspan="2" align="center">dev-clean</td>
    <td colspan="2" align="center">dev-other</td>
  </tr>

  <tr>
    <td>Model</td>
    <td>Layer</td>
    <td>within</td>
    <td>across</td>
    <td>within</td>
    <td>across</td>
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

# sSIMI : Semantic similarity with human judgments

In this task, the model receives 2 words, let's say /abduct/ and /kidnap/. 
Distance between the embeddings of these 2 words are computed. 
Then, a Spearman's rank correlation coefficent is computed between these distances, and human semantic similarity judgements.
Higher is better.

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="2" align="center">sSIMI dev</td>
  </tr>

  <tr>
    <td>Model</td>
    <td>Layer</td>
    <td align="center">librispeech</td>
    <td align="center">synthetic</td>
  </tr>

  <tr>
    <td>MFCCs+VG</td>
    <td>att</td>
    <td>11.8885</td>
    <td>6.3074</td>
  </tr>

  <tr>
    <td>CPC small+VG</td>
    <td>att</td>
    <td>13.0894</td>
    <td>9.4661</td>
  </tr>
</table>

# sBLIMP : Syntax acceptability judgment

In this task, the model receives two sentences, one of which is syntactically wrong. Let's say /dogs eat meat/ vs /dogs eats meat/.
The model is asked to return pseudo-probabilities for each of these sentence. The pseudo-probability of the syntactically right sentence is expected to be higher than the pseudo-probability of the syntactically wrong sentence.
Note that, extracting pseudo-probabilities from the model is part of modeling the task and can greatly impact the performance.
Higher is better.

<table>
  <tr>
    <td>Model</td>
    <td>K</td>
    <td>M_d</td>
    <td>Delta_t</td>
    <td colspan="1" align="center">sBLIMP dev</td>
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

# sWUGGY : Spot-the-word task

In this task, the model receives a word and a non-word. Let's say /brick/ and /blick/.
It it asked to return pseudo-probabilities for each of these word. The pseudo-probability of the word is expected to be higher 
than the pseudo-probability of the non-word.
As in the syntax acceptability judgment task, extracting pseudo-probabilities is part of modeling the task !
Higher is better.


<table>
  <tr>
    <td>Model</td>
    <td>K</td>
    <td>M_d</td>
    <td>Delta_t</td>
    <td colspan="1" align="center">sWUGGY dev</td>
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

# Findings

Overall, using the visual modality for learning speech representations in an unsupervised way seems on par with audio only models.

The ABX error rate obtained by CPC small is further improved with the VG model : it goes from 6.24% ABX error rate to 5.36% ABX error rate (librispeech dev-clean, within speakers).

The best achievement has been obtained on the semantic similarity task for which our best VG model got 13.09% as compared to 8.72% for the audio-only baseline (results reported here are computed on sSIMI librispeech dev set).

No improvement has been observed on the syntax acceptability judgment task (sBLIMP). A small decrease has been observed on the spot-the-word task : 70.69% for the audio-only basleine as compared to 67.16% for the multimodal baseline.