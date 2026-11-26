# Split data ipipnto training and testing sets
import numpy as np
import pandas as pd
from pathlib import Path
path  = ("data.csv") 
def load_csv(path):
    col_names = ['index','type', 'radius', 'texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry', 'fractal_dimension']

    df  = pd.read_csv(path,header=None ,names=col_names )
    # df  = df.rename(columns={0:'id', 1:'type'})
    # df.col_names
    # df =  pd.DataFrame(df, columns=['index'])
    # df.columns["name"]
    print(df.head(1))


load_csv(path)