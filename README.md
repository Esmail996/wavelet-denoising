# Signal denoising with Wavelets

This repository contains a signal denoising using the Wavelet's
multilevel decomposition. The current implementation is based on Python's 
package [PyWavelets](https://pywavelets.readthedocs.io/en/latest/).

## Main algorithm

  1. Preprocess the input signal S by removing any trends (like DC currents) 
  and normalizing it into the interval [0, 1]. The normalization is optional,
  and it doesn't happen automatically.
  2. Apply a multilevel wavelet decomposition on signal S using the method
  **wavedec** of PyWavelets.
  3. Use the detail coefficients cD from step 2 and determine the appropriate
  threshold value. Five different methods can be used to determine the threshold.
  4. Use the determined threshold value to apply soft or hard thresholding on
  the detail coefficients.
  5. Reconstruct the signal from the thresholded detail coefficients using the
  function **waverec** of PyWavelets. In case a normalization took place on input
  signal S at the preprocessing stage, renormalize.

## Threshold methods supported

There are five methods for determining the threshold so far. These methods are:
  1. **universal** The threshold, in this case, is given by the formula MAD x sqrt{2 x log(m)},
   where MAD is the Median Absolute Deviation, and m is the length of the signal.
  2. **sqtwolog** Same as the universal, except that it does not use the MAD.
  3. **energy** In this case, the thresholding algorithm estimates the energy levels
   of the detail coefficients and uses them to estimate the optimal threshold.
  4. **stein** This method implements Stein's unbiased risk estimator.
  5. **heurstein** This is a heuristic implementation of Stein's unbiased risk estimator.

> :rotating_light: When one uses the *stein*, *heurstein*, and *sqtwolog* they must
enable the rescale if the input signal does not have white noise (set the 
argument *selected_level=1* or *selected_level=nlevel*, where *nlevel* is the
Wavelet decomposition level).

> Both the *stein* and *heurstein* methods are implemented according to 
PyYAWT package (see [here](https://pyyawt.readthedocs.io/pyyawt.html#module-pyyawt.denoising)).

## Dependencies

**wavelet_denoising** requires the following packages:
  * Numpy
  * PyWavelets
  * Scipy
  * Sklearn
  * Matplotlib

You can install all the dependencies by typing the following in your terminal:
```bash
$ pip (or pip3) install -r requirements.txt
`

points to speak about:

bandwidth choice 
sampling Frequency
decomposition levels
wavelet family choice
decimation


