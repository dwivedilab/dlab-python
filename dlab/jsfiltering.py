import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

pd.options.mode.chained_assignment = None

def load_excel_file(filename, sheetname):
    if os.path.isfile(filename):
        xl = pd.ExcelFile(input_file)
    else:
        raise ValueError('Provided file "%s" cannot be found' % filename)
    
    if sheetname in xl.sheet_names:
        return xl.parse(sheetname)
    else:
        raise ValueError('Provided sheet "%s" cannot be found' % sheetname)

def load_excel_files(dir_path = ''):
    #first ask for the source data
    files = {}
    i = 1
    if dir_path == '':
        pass # put in desktop file dialog here
    print('Listing files...')
    for file in os.listdir(dir_path):
        if file.endswith('.xlsx'):
            files[i] = file
            print('%i: %s' % (i,file))
            i += 1
    print()

    input_file = input('Which file?: ')
    if int(input_file) in files.keys():
        filename = files[int(input-file)]

    #load excel file
    xl = pd.ExcelFile(filename)

    #ask which sheet the data is stored in
    sheets = {}
    i = 1
    print('\nListing sheets...')
    for sheet in xl.sheet_names:
        sheets[i] = sheet           
        print('%i: %s' % (i, sheet))
        i += 1
    print()

    input_sheet = input('Which sheet?: ')
    if int(input_sheet) in sheets.keys():
        sheetname = sheets[int(input_sheet)]
    
    #tell user how they may directly call this import
    print("You may call: load_excel_file('%s','%s')" % (filename,sheetname))
    
    #create pandas df from sheet
    return xl.parse(sheetname)

def check_input_data(df):
    cols = df.columns
    print("\nThe following columns are in the input data: %s" % cols)
    
    missing = []
    for col in ['Subject','item','condition','Trial','RT']:
        if not col in cols:
            missing.append(col)
    if missing:
        raise ValueError('Missing Column(s): "%s"' % missing)
    
    ppt = df['Subject'].unique()
    print("\nThere are %s ppts in this file." % len(ppt))
    
    conds = df['condition'].unique()

    if len(conds) > 1:
        print("\nThe following conditions are in this file:")
        for cond in conds:
            print(cond)
        
    elif len(conds) == 1:
        print("\nThere is only one condition in this file.")
        print(conds)
        
    else:
        raise ValueError("no conds")

    print("\nThere are %s records." % len(df.index))
    

def remove_extremes(df, low = 200, high = 5000, summarize = True):
    #filter out below 200 and above 5000
    df['RTremove'] = pd.cut(df['RT'], [0, low, high, np.inf], labels=['Below','Ok','Above'])
    removeExtremes = (df['RT'] < high)&(df['RT'] > low)
    df['RTremoveVal'] = np.nan
    df['RTremoveVal'][removeExtremes] = df['RT'][removeExtremes]

    if summarize:
        total = len(df.index)
        removed = df['RTremove'].value_counts()
        print("\nThe summary for removal of extreme values:")
        print(">>%s or %.3f%% items were below the specified cutoff of %s." % (removed['Below'], removed['Below']*100/total, low))
        print(">>%s or %.3f%% items were above the specified cutoff of %s." % (removed['Above'], removed['Above']*100/total, high))

    return df

def identify_missing_data(df):
    by_ppt = df.groupby(['Subject','condition','Trial'])
    missing_by_ppt = by_ppt.mean()[by_ppt.mean().isnull().any(axis=1)]['RTremoveVal']
    if len(missing_by_ppt) != 0:
        print('\nMissing data if filtered by ppt:\n%s\n' % missing_by_ppt)

        inelig_ppts = missing_by_ppt.index.get_level_values('Subject').unique()
        remove = input("""Ineligible Participants:\n\n  %s \n\nRemove these participants? (y/n): """ % list(inelig_ppts))
        if remove == 'y':
            df = df[~df['Subject'].isin(inelig_ppts)]
            print('\nRemoved!')
        
    by_item = df.groupby(['item','condition','Trial'])
    missing_by_item = by_item.mean()[by_item.mean().isnull().any(axis=1)]['RTremoveVal']
    if len(missing_by_item) != 0:
        print('\nMissing data if filtered by item:\n%s' % missing_by_item)

    return df

def filter_outliers(df, ppt = True, items = True, SD = 2, summarize = True):
    total = len(df.index)
    def outliers(group, labels_name, trimmed_name):
        mean, std = group.mean(), group.std()
        if np.isnan(std):
            std = 0.1
        lower, upper = mean - SD*std, mean + SD*std
        trimmed = group.mask(group < lower, lower).mask(group > upper, upper)
        labels = pd.cut(group, [np.NINF, lower, upper, np.inf], labels=['Below','Ok','Above'])
        return pd.DataFrame({trimmed_name:trimmed, labels_name:labels})

    #filter out outliers by ppt then by items
    if ppt:
        df[['RTfiltered_by_ppt_labels','RTfiltered_by_ppt_values']] = df.groupby(['Subject','condition','Trial'])['RTremoveVal'].apply(outliers,'RTfiltered_by_ppt_labels','RTfiltered_by_ppt_values')

        if summarize:
            by_ppt_filtered = df['RTfiltered_by_ppt_labels'].value_counts()
            print("\nThe summary for filtering by ppt:")
            print(">>%s or %.3f%% items were below the specified cutoff of -%sSD." % (by_ppt_filtered['Below'], by_ppt_filtered['Below']*100/total, SD))
            print(">>%s or %.3f%% items were above the specified cutoff of +%sSD." % (by_ppt_filtered['Above'], by_ppt_filtered['Above']*100/total, SD))
                    
    if items:
        df[['RTfiltered_by_item_labels','RTfiltered_by_item_values']] = df.groupby(['item','condition','Trial'])['RTremoveVal'].apply(outliers,'RTfiltered_by_item_labels','RTfiltered_by_item_values')

        if summarize:
            by_item_filtered = df['RTfiltered_by_item_labels'].value_counts()
            print("\nThe summary for filtering by items:")
            print(">>%s or %.3f%% items were below the specified cutoff of -%sSD." % (by_item_filtered['Below'], by_item_filtered['Below']*100/total, SD))
            print(">>%s or %.3f%% items were above the specified cutoff of +%sSD." % (by_item_filtered['Above'], by_item_filtered['Above']*100/total, SD))

    return df

def save_df(df):
    print('Note: The specified file will be overwritten.')
    output_file = input('Name of your excel output file: ')
    if output_file[-5:] != '.xlsx':
        output_file += '.xlsx'
        
    writer = pd.ExcelWriter(output_file)
    df.to_excel(writer,'Filtered Data')
    writer.save()

def plot_reading_time(df, factor_col, conds, formatting, words, title, adj_factor = .05, e_cap = 2, e_width = 1, e_c = 'black', lw = 1, mk = 5):
    df = df.groupby(['Subject',factor_col,'Trial'])['RTfiltered_by_ppt_values'].mean()
    trials = df.index.get_level_values('Trial').unique()
    
    x = {}
    for i in range(len(conds)):
        adj = (i - (len(conds) - 1)/2) * (adj_factor)
        x[conds[i]] = []
        for j in range(len(trials)):
            x[conds[i]].append(j + 1 + adj)
    
    y = df.groupby([factor_col,'Trial']).mean().unstack()
    yerr = df.groupby([factor_col,'Trial']).sem().unstack()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cond in conds:
        c, fmt = formatting[cond]['c'], formatting[cond]['fmt']
        ax.errorbar(x[cond], y.loc[cond],yerr=yerr.loc[cond],
                     c=c, fmt=fmt, label = cond,
                     ecolor= e_c, capsize = e_cap, elinewidth = e_width,
                     lw = lw, markersize = mk)
        
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in range(len(trials) + 1):
            return words[int(tick_val) - 1]
        else:
            return ''
    
    ax.xaxis.set_major_locator(FixedLocator(range(len(trials) + 1)))
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.tick_params('x', labelrotation = 45)

    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Region', weight = 'bold', size = 'x-large')
    plt.ylabel('Reading Time (ms)', weight = 'bold', size = 'x-large')
    
    ax.legend()
    plt.show()

QScope_Defaults = {'Context_Number': {'AP':{'c':'red','fmt':'-^'},
                                      'AS':{'c':'orange','fmt':'-^'},
                                      'CP':{'c':'green','fmt':'-s'},
                                      'CS':{'c':'blue','fmt':'-s'}},
                   'Context':{'Control':{'c':'blue','fmt':'-s'},
                              'Ambiguous':{'c':'red','fmt':'-^'}},
                   'Number':{'Singular':{'c':'orange','fmt':'-s'},
                             'Plural':{'c':'purple','fmt':'-^'}}
                   }

QScope_words = {'QBehQ':{'S1_S2':['Every','kid','climbed','a/that/those','tree(s).','The','tree(s)','was/were', 'in', 'the', 'park.'],
                         'S1':['Every','kid','climbed','a/that/those','tree(s).'],
                         'S2':['The','tree(s)','was/were', 'in', 'the', 'park.']},
                'QBehQ4070':{'S1_S2':['Every','jeweller','appraised','a/that/those','diamond(s).','The','diamond(s)','was/were', 'clear', 'and', 'flawless.'],
                             'S1':['Every','jeweller','appraised','a/that/those','diamond(s).'],
                             'S2':['The','diamond(s)','was/were', 'clear', 'and', 'flawless.']}}
