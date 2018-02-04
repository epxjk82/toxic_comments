import pandas as pd
import numpy as np
import operator

test_dist_d = {'toxic': 0.182379,
               'severe_toxic':0.014615,
               'obscene':0.114349,
               'threat': 0.002984,
               'insult': 0.076583,
               'identity_hate': 0.013819}

def get_eval_data(train_df, list_classes, val_split, test_dist_dict=test_dist_d):

    val_size = np.ceil(train_df.shape[0]*val_split)

    print ("Creating validation set with size {}...".format(val_size))

    test_dist_n_d = {}

    for col in list_classes:
        test_dist_n_d[col] = int(test_dist_d[col]*val_size)

    train_tmp_df = train_df.copy()

    n_threat_val = int(val_size*test_dist_d['threat'])

    val_custom_df = train_tmp_df[train_tmp_df['threat']==1].sample(n=n_threat_val)
    val_custom_df[list_classes].sum(axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_identity_hate_val = int(val_size*test_dist_d['identity_hate']) - val_custom_df['identity_hate'].sum()
    val_identity_hate_df = train_tmp_df[np.array(train_tmp_df['identity_hate']==1) & \
                                        np.array(train_tmp_df['threat']!=1) ].sample(n=n_identity_hate_val)
    val_identity_hate_df[list_classes].sum(axis=0)

    val_custom_df = pd.concat([val_custom_df, val_identity_hate_df], axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_severe_toxic_val = int(val_size*test_dist_d['severe_toxic']) - val_custom_df['severe_toxic'].sum()
    val_severe_toxic_df = train_tmp_df[np.array(train_tmp_df['severe_toxic']==1) & \
                                       np.array(train_tmp_df['identity_hate']!=1) & \
                                       np.array(train_tmp_df['threat']!=1)].sample(n=n_severe_toxic_val)
    val_severe_toxic_df[list_classes].sum(axis=0)

    val_custom_df = pd.concat([val_custom_df, val_severe_toxic_df], axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_insult_val = int(val_size*test_dist_d['insult']) - val_custom_df['insult'].sum()
    val_insult_df = train_tmp_df[np.array(train_tmp_df['insult']==1)  & \
                                  np.array(train_tmp_df['severe_toxic']!=1) & \
                                  np.array(train_tmp_df['identity_hate']!=1) & \
                                  np.array(train_tmp_df['threat']!=1)].sample(n=n_insult_val)
    val_insult_df[list_classes].sum(axis=0)

    val_custom_df = pd.concat([val_custom_df, val_insult_df], axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_obscene_val = int(val_size*test_dist_d['obscene']) - val_custom_df['obscene'].sum()
    val_obscene_df = train_tmp_df[np.array(train_tmp_df['obscene']==1) & \
                                  np.array(train_tmp_df['severe_toxic']!=1) & \
                                  np.array(train_tmp_df['insult']!=1) & \
                                  np.array(train_tmp_df['identity_hate']!=1) & \
                                  np.array(train_tmp_df['threat']!=1)].sample(n=n_obscene_val)
    val_obscene_df[list_classes].sum(axis=0)

    val_custom_df = pd.concat([val_custom_df, val_obscene_df], axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_toxic_val = int(val_size*test_dist_d['toxic']) - val_custom_df['toxic'].sum()
    val_toxic_df = train_tmp_df[np.array(train_tmp_df['toxic']==1) & \
                                 np.array(train_tmp_df['obscene']!=1) & \
                                  np.array(train_tmp_df['severe_toxic']!=1) & \
                                  np.array(train_tmp_df['insult']!=1) & \
                                  np.array(train_tmp_df['identity_hate']!=1) & \
                                  np.array(train_tmp_df['threat']!=1)].sample(n=n_toxic_val)
    val_toxic_df[list_classes].sum(axis=0)

    val_custom_df = pd.concat([val_custom_df, val_toxic_df], axis=0)
    val_custom_df[list_classes].sum(axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_tmp_df = train_df.copy().iloc[not_ind]

    n_noflag_val = int(val_size - val_custom_df.shape[0])

    val_noflag_df = train_tmp_df[np.array(train_tmp_df['toxic']!=1) & \
                                 np.array(train_tmp_df['obscene']!=1) & \
                                  np.array(train_tmp_df['severe_toxic']!=1) & \
                                  np.array(train_tmp_df['insult']!=1) & \
                                  np.array(train_tmp_df['identity_hate']!=1) & \
                                  np.array(train_tmp_df['threat']!=1)].sample(n=n_noflag_val)

    val_custom_df = pd.concat([val_custom_df, val_noflag_df], axis=0)

    not_ind = np.setdiff1d(train_df.index, val_custom_df.index)
    train_custom_df = train_df.copy().iloc[not_ind]
    assert train_df.shape[0]==train_custom_df.shape[0] +val_custom_df.shape[0]

    print ("Created train set with shape ", train_custom_df.shape)
    print ("---------------------------------")
    print ("Train set distribution :")
    print (train_custom_df[list_classes].mean())
    print ("")
    print ("Created validation set with shape ", val_custom_df.shape)
    print ("---------------------------------")
    print ("Validation Set distribution :")
    print (val_custom_df[list_classes].mean())

    return train_custom_df, val_custom_df
