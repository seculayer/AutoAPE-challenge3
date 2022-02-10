import eli5

def Feature_Select(model):
    print(eli5.formatters.as_dataframe.explain_weights_df(model).feature)
    top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature if 'BIAS' not in i]
    return top_features