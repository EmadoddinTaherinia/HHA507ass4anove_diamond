# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:29:20 2021

@author: Emad
"""

import pandas as pd
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kurtosis
from scipy.stats import skew, bartlett
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm

### dataset dictionary
# A data frame with 53940 rows and 10 variables:
# price price in US dollars (\$326--\$18,823)
# carat weight of the diamond (0.2--5.01)
# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# color diamond colour, from J (worst) to D (best)
# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# x length in mm (0--10.74)
# y width in mm (0--58.9)
# z depth in mm (0--31.8)
# depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# table width of top of diamond relative to widest point (43--95)

diamond = pd.read_csv('https://raw.githubusercontent.com/EmadoddinTaherinia/HHA507ass4anove_diamond/main/diamonds.csv')

# Factor1; cut with 5 levels
diamond.cut.value_counts()
diamond['cut'].value_counts()
len(diamond.cut.value_counts())

# Factor2; clarity with 8 levels
diamond.clarity.value_counts()
diamond['clarity'].value_counts()
len(diamond['clarity'].value_counts())

# Factor3; color with 7 levels
diamond.color.value_counts()
diamond['color'].value_counts()
len(diamond['color'].value_counts())

## continuous value:
    
price = diamond['price']


# 1-way anova
# 1 DV  price
# 1 IV  cut


# question1: Is there a difference between Ideal, Good and Fair levels of cut and price?
# P-value < 0.05 indicates that there is a dsignificance difference between cut level and price
# To clarify the difference between diferrent cuts we can run tukey test.

model1 = smf.ols("price ~ C(cut)", data = diamond).fit()
stats.shapiro(model1.resid)
sm.stats.anova_lm(model1)

comp = mc.MultiComparison(diamond['price'], diamond['cut'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())


cut1 = diamond[diamond['cut'] == 'Ideal']
cut2 = diamond[diamond['cut'] == 'Good']
cut3 = diamond[diamond['cut'] == 'Fair']

plt.hist(cut1['price'])
plt.show

plt.hist(cut2['price'])
plt.show

plt.hist(cut3['price'])
plt.show

# kurtosis

kurtosis(cut1['price'])
kurtosis(cut2['price'])
kurtosis(cut3['price'])

# skewness

skew(cut1['price'])
skew(cut2['price'])
skew(cut3['price'])

# homogeneity of variance, bartlett test

stats.bartlett(cut1['price'],
               cut2['price'],
               cut3['price'])

stats.f_oneway(cut1['price'],
               cut2['price'],
               cut3['price'])



### question2; Is there a difference between SI1, VS2 and VVS1 levels of clarity and price?
# P-value < 0.05 indicates that there is a dsignificance difference between clarity level and price
# To clarify the difference between diferrent clarity we can run tukey test.

model2 = smf.ols("price ~ C(clarity)", data = diamond).fit()
stats.shapiro(model2.resid)
sm.stats.anova_lm(model2) 

comp = mc.MultiComparison(diamond['price'], diamond['clarity'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())

diamond.clarity.value_counts()

clarity1 = diamond[diamond['clarity'] == 'SI1']
clarity2 = diamond[diamond['clarity'] == 'VS2']
clarity3 = diamond[diamond['clarity'] == 'VVS1']


plt.hist(clarity1['price'])
plt.show

plt.hist(clarity2['price'])
plt.show

plt.hist(clarity3['price'])
plt.show

# kurtosis

kurtosis(clarity1['price'])
kurtosis(clarity2['price'])
kurtosis(clarity3['price'])

# skewness

skew(clarity1['price'])
skew(clarity2['price'])
skew(clarity3['price'])

# homogeneity of variance, bartlett test

stats.bartlett(clarity1['price'],
               clarity2['price'],
               clarity3['price'])

stats.f_oneway(clarity1['price'],
               clarity2['price'],
               clarity3['price'])




# question3: Is there a difference between G, H and I levels of color and price?
# P-value < 0.05 indicates that there is a dsignificance difference between color level and price
# To clarify the difference between diferrent color we can run tukey test.

model3 = smf.ols("price ~ C(color)", data = diamond).fit()
stats.shapiro(model3.resid)
sm.stats.anova_lm(model3)

comp = mc.MultiComparison(diamond['price'], diamond['color'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())


    
color1 = diamond[diamond['color'] == 'G']
color2 = diamond[diamond['color'] == 'H']
color3 = diamond[diamond['color'] == 'I']



plt.hist(color1['price'])
plt.show

plt.hist(color2['price'])
plt.show

plt.hist(color3['price'])
plt.show

# kurtosis

kurtosis(color1['price'])
kurtosis(color2['price'])
kurtosis(color3['price'])
 
 # skewness

skew(color1['price'])
skew(color2['price'])
skew(color3['price'])

# homogeneity of variance, bartlett test

stats.bartlett(color1['price'],
               color2['price'],
               color3['price'])

stats.f_oneway(color1['price'],
               color2['price'],
               color3['price'])

# Answer: for all 3 levels of cuts, clarity and color investigated in this research skewness is greater than 1 which means we have a positive skew,
# and kurtosis values are also over o that shows the presence of potential outliers (leptokurtic)