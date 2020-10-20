# Credit Approval Application
## Overview
A credit approval application which decides if a bank client should get credit or not. KNN and KMeans algorithms are used and implemented in C-language. For the validation of solution, K-fold cross validation is used.

Dataset concerns credit card applications.  All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

## About Dataset
- Number of Instances: 690
- Number of Attributes: 15 + class attribute
- Attribute Information:
    - A1:	b, a.
    - A2:	continuous.
    - A3:	continuous.
    - A4:	u, y, l, t.
    - A5:g, p, gg.
    - A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    - A7:	v, h, bb, j, n, z, dd, ff, o.
    - A8:	continuous.
    - A9:	t, f.
    - A10:	t, f.
    - A11:	continuous.
    - A12:	t, f.
    - A13:	g, p, s.
    - A14:	continuous.
    - A15:	continuous.
    - A16: +,-         **(class attribute)**

- Missing Attribute Values: 37 cases (5%) have one or more missing values.  The missing values from particular attributes are:
	- A1: 12
	- A2: 12
	- A4: 6
	- A5: 6
	- A6: 9
	- A7: 9
	- A14: 13

- Class Distribution: 
	- +: 307 (44.5%)
	- -: 383 (55.5%)

## Dataset Resources
- https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names
- https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data

## Credits
  - Dataset submitted by quinlan@cs.su.oz.au
