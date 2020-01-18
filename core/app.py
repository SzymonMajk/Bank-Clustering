import pandas as pd
import numpy as np


class Core:

    def cluster(self, method):
        pass

    def get_results(self, xaxis, yaxis):

        #number of points per group
        n = 50

        #define group labels and their centers
        groups = {'A': (2,2),
            'B': (3,4),
            'C': (4,4),
            'D': (4,1)}

        #create labeled x and y data
        data = pd.DataFrame(index=range(n*len(groups)), columns=['x','y','label'])
        for i, group in enumerate(groups.keys()):
            #randomly select n datapoints from a gaussian distrbution
            data.loc[i*n:((i+1)*n)-1,['x','y']] = np.random.normal(groups[group], 
                                                           [0.5,0.5], 
                                                           [n,2])
            #add group labels
            data.loc[i*n:((i+1)*n)-1,['label']] = group
        return data