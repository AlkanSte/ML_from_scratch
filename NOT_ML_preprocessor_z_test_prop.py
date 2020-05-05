mport pandas as pd
import numpy as np

#Importing data, viewing top
result = pd.read_csv('kniga1.csv', encoding="windows-1251", sep = ';')
#print(result.head(3))
site = pd.read_csv('url_site.csv', encoding="windows-1251", sep = ';', header = None)
site = pd.DataFrame(site)
#print(site.head(3))
forum = pd.read_csv('url_forum.csv', encoding="windows-1251", sep = ';', header = None)
forum = pd.DataFrame(forum)
#print(forum.head(3))

#Drop duplicate rows, clean NA
result = result.drop_duplicates(subset =["id", 'Telephone'])
site = site.drop_duplicates()
forum = forum.drop_duplicates()
result = result.dropna(subset=['Telephone'])



#Present conversion rates. This could be easily coded for any kpi
Converse = [{'Basic': len(result.Telephone.unique()) / len(site),
         #'Forum': len(result_forum)/len(result.Telephone.unique()),
         'Forum': len(result.Telephone.unique())/forum[0].nunique(),
         'IS_THIS_HOT': len(result.Telephone.unique())/len(forum_site)}]
Converse = pd.DataFrame(Converse)

print('Conversion rates')
print(Converse)

#Make some arrays for z proportion test
result_site = site[site[0].isin(result.Telephone)]
forum_site = forum[forum[0].isin(site[0])]
result_forum = forum[forum[0].isin(result.Telephone)]
result_both = forum_site[forum_site[0].isin(result.Telephone)]

#Construct binary arrays of successes 
basic1 = np.zeros(shape=(len(site) - len(result_site),1))
basic2 = np.zeros(shape=(len(result_site),1)) + 1
basic3 = np.concatenate((basic1, basic2))
forum1 = np.zeros(shape=(len(forum_site) - len(result_both),1))
forum2 = np.zeros(shape=(len(result_both),1)) + 1
forum3 = np.concatenate((forum1, forum2))

#Start testing
from statsmodels.stats.proportion import proportions_ztest
count = np.array([np.count_nonzero(basic3 == 1), np.count_nonzero(forum3 == 1)])
nobs = np.array([len(basic3), len(forum3)])
stat, pval = proportions_ztest(count, nobs)

print('\n')
print('z-proportion test p-value')
print('{0:0.3f}'.format(pval))
#If p-value is lower than our threshold, then we did it! We got new kpi
