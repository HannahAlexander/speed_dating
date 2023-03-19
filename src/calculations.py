import umap
import pandas as pd

def umap_transformation(df): 
    '''
    Genertaes a umap cluster plot 

    Arguments:
    df- dataset containing leads to be used 

    Returns:
    The clustered leads

    '''
    standard_embedding = umap.UMAP(random_state=42, n_neighbors=50, min_dist=0.01, metric='canberra').fit(df) 
    return standard_embedding.embedding_

def full_summary(clustered_df):
    '''
    Genrates a table of results where each profile has an associated value for each feature 

    Arguments:
    clustered_df - dataset which contian the leads with their feature information and cluster label 

    Returns:
    df_profile - the full dataframe containing all the resultant feature values 

    '''

    # Overall level summary
    df_profile_overall = clustered_df.describe().T

    # using mean; use appropriate summarization (median, count, etc.) for each feature
    df_profile_overall['Overall Dataset'] = df_profile_overall[['mean']]
    df_profile_overall = df_profile_overall[['Overall Dataset']]

    # Cluster ID level summary
    df_cluster_summary = clustered_df.groupby('labels').describe().T.reset_index()
    df_cluster_summary = df_cluster_summary.rename(columns={'level_0':'column','level_1':'metric'})

    # using mean; use appropriate summarization (median, count, etc.) for each feature
    df_cluster_summary = df_cluster_summary[df_cluster_summary['metric'] == "mean"] #mean
    df_cluster_summary = df_cluster_summary.set_index('column')

    # join into single summary dataset
    df_profile = df_cluster_summary.join(df_profile_overall) # joins on Index

    return df_profile

def best_cluster_results(cluster, count, catagorical, df_profile):
    '''
    Finds the most common value for a catagorical feature 

    Arguments:
    cluster - the cluster label 
    count - the amount of leads in the cluster 
    catagorical - the list of catagorical values to consider
    df_profile - the dataframe contianing the leads and their associated cluster label  

    Returns:
    results - the list of catagorical features with the resultant most common value for the profile 

    '''
    results = []
    #results = pd.DataFrame()
    for cat in catagorical:
        cat_var = cat
        best_value = 0
        best_row = ""
        for row in range(df_profile.shape[0]):
            if cat in df_profile.index[row]:
                if ((df_profile.iloc[row,cluster+2]*count) > best_value):
                    best_value = (df_profile.iloc[row,cluster+2]*count)
                    best_row = df_profile.index[row].replace(cat, '')
        results.append([cat_var, best_row, best_value])
        #res = pd.DataFrame([[best_row, best_value]], columns=["name", "frequency"])
        #results(res)
    return results

def column(matrix, i):
    '''
    Returns the desired column in a matrix 

    Arguments:
    matrix - the matrix to be used 
    i - the column to be retrieved 

    Returns:
    The desired column in the matrix 

    '''
    return [row[i] for row in matrix]

def best_profiles(df_count_p, continuous, catagorical, df_profile):
    '''
    Generates the profile with the most common features for the cluster 

    Arguments:
    df_count_p - the data frame containing the clustering segments 
    continuous - the list of continuous values to consider
    catagorical - the list of catagorical values to consider
    df_profile - the dataframe contianing the leads and their associated cluster label  

    Returns:
    results_table - a dataframe contianing all the profile's features with the resultant most common value

    '''
    
    labels = [i for i in df_count_p.index]
    counts = df_count_p['Total']

    all_cont=[]
    for label in labels:
        cont_vals = []
        for cont in continuous:
            val = df_profile.loc[cont, label]
            cont_vals.append([cont, val])
        all_cont.append([label, cont_vals]) 

    all = []
    for label in labels:
        all.append([label, best_cluster_results(label, counts[label], catagorical, df_profile)])

    vals_cont = [column(all_cont[i][1],1) for i in range(len(labels))]
    names_cat = [column(all[i][1],1) for i in range(len(labels))]

    vals = [(vals_cont[i] + names_cat[i]) for i in range(len(vals_cont))]
    results_table = pd.DataFrame(vals, columns = [continuous + catagorical], index = labels)
    
    return results_table