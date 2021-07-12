#%%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
# draw cifar 10 model
for type in ['plain','residual']:
    df_train = pd.DataFrame(columns=['iter','model','error'])
    dataset = 'train'
    for model in ['res20','res32','res44','res56']:
        file_path = '../logs/cifar10/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 50
        else:
            temp['iter'] = temp['iter'] * 354

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_train = df_train.append(temp)

    model = 'res110'
    for i in range(1,6):
        file_path = '../logs/cifar10/residual/'+model+'_residual_'+dataset+'_err'+str(i)+'.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 50
        else:
            temp['iter'] = temp['iter'] * 354

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_train = df_train.append(temp)
        
    # 过滤非整epoch
    df_train = df_train[df_train['iter'] % 2000 ==0]
    df_train['dataset'] = 'train'
    df_train['error'] = df_train['error']*100

    df_test = pd.DataFrame(columns=['iter','model','error'])
    dataset = 'test'
    for model in ['res20','res32','res44','res56']:
        file_path = '../logs/cifar10/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 50
        else:
            temp['iter'] = temp['iter'] * 354

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_test = df_test.append(temp)

    model = 'res110'
    for i in range(1,6):
        file_path = '../logs/cifar10/residual/'+model+'_residual_'+dataset+'_err'+str(i)+'.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 50
        else:
            temp['iter'] = temp['iter'] * 354

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_test = df_test.append(temp)
        

    # 过滤非整epoch
    df_test = df_test[df_test['iter'] % 4 ==0]
    df_test['error'] = df_test['error']*100
    df_test['dataset'] = 'val'
    # sns.lineplot(x='iter',y='error',hue='model',style='model',data=df_test, dashes=[0.1,0.])
    # df = df_train.append(df_test)
    df = df_test.append(df_train)

    plt.figure(figsize=(8, 5))
    if type == 'plain':
        plt.ylim(0, 70)
    else:
        plt.ylim(0, 30)
    sns.lineplot(x='iter',y='error',hue='model',style='dataset',data=df)
    

# %%
# draw tiny_imagenet model
# model_list = ['res18','res34']
model_list = ['res18','res34','res50','res101']
for type in ['plain','residual']:
    df_train = pd.DataFrame(columns=['iter','model','error'])
    dataset = 'train'
    for model in model_list:
        file_path = '../logs/tiny_imagenet/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 35
            pass
        else:
            temp['iter'] = temp['iter'] * 250

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_train = df_train.append(temp)
    
    if 'res101' in model_list and type == 'residual':
        model = 'res152'
        file_path = '../logs/tiny_imagenet/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 35
            pass
        else:
            temp['iter'] = temp['iter'] * 250

        temp['model'] = model
        df_train = df_train.append(temp)
        
    # 过滤非整epoch
    df_train = df_train[df_train['iter'] % 100 ==0]
    df_train['dataset'] = 'train'
    df_train['error'] = df_train['error']*100

    df_test = pd.DataFrame(columns=['iter','model','error'])
    dataset = 'test'
    for model in model_list:
        file_path = '../logs/tiny_imagenet/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 35
        else:
            temp['iter'] = temp['iter'] * 250

        if type == 'residual':
            temp['model'] = model
        else:
            temp['model'] = type + model.replace('res','')
        df_test = df_test.append(temp)

    if 'res101' in model_list and type == 'residual':
        model = 'res152'
        file_path = '../logs/tiny_imagenet/'+type+'/'+model+'_'+type+'_'+dataset+'_err.csv'
        temp = pd.read_csv(file_path,names=['iter','unknown','error'])
        temp = temp.drop(columns=['unknown'],axis=1)
        if dataset == 'train':
            temp['iter'] = temp['iter'] * 35
        else:
            temp['iter'] = temp['iter'] * 250

        temp['model'] = model
        df_test = df_test.append(temp)
        
    # 过滤非整epoch
    df_test = df_test[df_test['iter'] % 4 ==0]
    df_test['error'] = df_test['error']*100
    df_test['dataset'] = 'val'

    df = df_test.append(df_train)

    plt.figure(figsize=(8, 5))
    if type == 'plain':
        plt.ylim(0, 100)
    else:
        plt.ylim(0, 100)
    sns.lineplot(x='iter',y='error',hue='model',style='dataset',data=df)
    plt.show()
    
# %%
