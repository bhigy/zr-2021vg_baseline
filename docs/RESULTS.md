
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


<table>
  <tr>
    <td>Model</td>
    <td>K</td>
    <td>M_d</td>
    <td>Delta_t</td>
    <td colspan="1" align="center">sBLIMP dev</td>
  </tr>


  <tr>
    <td>MFCCs+VG+KMEANS+BERT small</td>
    <td>50</td>
    <td>10</td>
    <td>1</td>
    <td>53.19</td>
  </tr>

  <tr>
    <td>CPC small+VG+KMEANS+BERT large</td>
    <td>500</td>
    <td>10</td>
    <td>1</td>
    <td>54.68</td>
  </tr>
</table>

<table>
  <tr>
    <td>Model</td>
    <td>K</td>
    <td>M_d</td>
    <td>Delta_t</td>
    <td colspan="1" align="center">sWUGGY dev</td>
  </tr>


  <tr>
    <td><pre>MFCCs+VG+
KMEANS+
BERT small</pre></td>
    <td>50</td>
    <td>10</td>
    <td>1</td>
    <td>52.53</td>
  </tr>

  <tr>
    <td><pre>CPC small+VG+
KMEANS+
BERT large</pre></td>
    <td>500</td>
    <td>10</td>
    <td>1</td>
    <td>67.16</td>
  </tr>
</table>