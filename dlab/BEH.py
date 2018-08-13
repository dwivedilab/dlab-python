import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

pd.options.mode.chained_assignment = None

class Project:
    """
    BEH.Project CLASS DOCSTRING
    """
    def __init__(self, data, load_configs = "", **kwargs):
        self.RTdata = None
        self.CompQdata = None
        self.plot_configs = plot_configs.get(load_configs, {})
        if not isinstance(self.plot_configs, dict):
            raise TypeError("self.plot_configs is of type: %s instead of dict." % type(self.plot_configs))

        required_columns = ['PPT','Condition','Item']
        if any(column not in kwargs for column in required_columns):
            raise ValueError("Ensure you have the columns: 'PPT', 'Condition', 'Item'")

        recommended_columns = ['WordPos','RT','CompQAcc','CompQRT']
        recommended_columns_not_found = []
        for column in recommended_columns:
            if column not in kwargs:
                recommended_columns_not_found.append(column)

        if recommended_columns_not_found:
            print("The following recommended columns columns were not found in **kwargs: %s" % ", ".join(recommended_columns_not_found))
            print("Note that these recommended columns may not be required depending on the design. However:")
            if 'CompQAcc' in recommended_columns_not_found or 'CompQRT' in recommended_columns_not_found:
                print('\tBecause CompQ columns were not provided, get_CompQdata(), plot_CompQAcc() and plot_CompQRT() will not work.')
            if 'WordPos' in recommended_columns_not_found or 'RT' in recommended_columns_not_found:
                print('\tBecause WordPos and RT columns were not provided, get_RTdata() and plot_reading_times() will not work.')

        df_columns_not_found = []
        for k,v in kwargs.items():
            if v not in data.columns:
                df_columns_not_found.append(v)

        if df_columns_not_found:
            raise ValueError("The following columns were not found in input data: %s" % ", ".join(df_columns_not_found))

        self.data = data.rename(columns={v: k for k, v in kwargs.items()})

    def __str__(self):
        return "A __str__ method has not yet been implemented"

    def compute_avgs(self, inputs, outputs, groupby = []):
        if self.data == None:
            raise ValueError("No data loaded in self.data")

        if isinstance(groupby, list):
            if groupby:
                if any(factor not in self.data.columns for factor in groupby):
                    raise ValueError("Provided groupby contains columns not found in self.data")
            else:
                groupby = ['PPT','Item']
                if 'WordPos' in self.data.columns:
                    groupby.append('WordPos')
        else:
            raise TypeError("Provided groupby must be a list. Not of type: %s" % type(groupby))
        output_df = self.get_conditions(inputs).groupby(groupby).mean()
        self.data = pd.concat([self.data, output_df], sort = False).reset_index()
        
    def get_RTdata(self, critical_conditions = [], summarize = True, remove_extremes = [200,5000], identify_missing_data = True, filter_outliers = 2):
        """
        columns: {'PPT':'Subject', 'WordPos':'Trial', 'Condition':'condition', 'List':'ExperimentName', 'Item':'item', 'RT':'ShowText.RT'}
        """
        def _remove_extremes(df, low = 200, high = 5000, summarize = True):
            df['RTremove'] = pd.cut(df['RT'], [0, low, high, np.inf], labels=['Below','Ok','Above'])
            removeExtremes = (df['RT'] < high)&(df['RT'] > low)
            df['RTremoveVal'] = np.nan
            df['RTremoveVal'][removeExtremes] = df['RT'][removeExtremes]

            if summarize:
                total = len(df.index)
                removed = df['RTremove'].value_counts()
                below = removed.get("Below", 0)
                above = removed.get("Above", 0)
                print("\nThe summary for removal of extreme values:")
                print(">>%s or %.3f%% items were below the specified cutoff of %s ms." % (below, below*100/total, low))
                print(">>%s or %.3f%% items were above the specified cutoff of %s ms." % (above, above*100/total, high))

            return df

        def _identify_missing_data(df, remove = True):
            by_ppt = df.groupby(['PPT','Condition','WordPos'])
            missing_by_ppt = by_ppt.mean()[by_ppt.mean().isnull().any(axis=1)]['RTremoveVal']
            if len(missing_by_ppt) != 0:
                print('\nMissing data if filtered by ppt:\n%s\n' % missing_by_ppt)
                

                if critical_conditions:
                    critical_conditions_in_data = []
                    for condition in critical_conditions:
                        if condition in missing_by_ppt.index.get_level_values('Condition').unique():
                            critical_conditions_in_data.append(condition)

                    if critical_conditions_in_data:
                        missing_by_ppt = missing_by_ppt.reset_index()
                        missing_by_ppt_crit = missing_by_ppt[missing_by_ppt['Condition'].isin(critical_conditions_in_data)].set_index(['PPT','Condition','WordPos'])
                        missing_by_ppt_filler = list(missing_by_ppt[~missing_by_ppt['Condition'].isin(critical_conditions_in_data)]['Condition'].unique())
                        inelig_ppts = missing_by_ppt_crit.index.get_level_values('PPT').unique()
                        print("The following ppts are missing data in critical conditions: %s" % (", ".join([str(x) for x in inelig_ppts])))
                        print("These ppts are being removed...")
                        df = df[~df['PPT'].isin(inelig_ppts)]
                        print('Removed!')
                        print('\nIf these ppts should be dropped, you should mark this on the run sheet & questionnaire data document and move the raw data to an inelig subfolder to prevent these ppts from being loaded again.')
                        if missing_by_ppt_filler:
                            print("\nThere is missing data in filler conditions.")
                            print("To avoid removing ppts with valid data in critical conditions, the following conditions will not be filtered: %s" % ", ".join(missing_by_ppt_filler))
                            df = df[~df['Condition'].isin(missing_by_ppt_filler)]
                    else:
                        print("Didn't need to drop any ppts as none of the above missing data affects provided critical_conditions")
                else:
                    print("No critical_conditions were provided so no ppts were dropped")
            else:
                print("No data missing by ppts.")

            by_item = df.groupby(['Item','Condition','WordPos'])
            missing_by_item = by_item.mean()[by_item.mean().isnull().any(axis=1)]['RTremoveVal']
            if len(missing_by_item) != 0:
                print('\nMissing data if filtered by item:\n%s' % missing_by_item)
                print("This should never happen - not sure what to do :(")

            return df
        
        def _filter_outliers(df, ppt = True, items = True, SD = 2, summarize = True):
            total = len(df.index)
            def outliers(group, labels_name, trimmed_name):
                mean, std = group.mean(), group.std()
                if np.isnan(std):
                    std = 0.1
                lower, upper = mean - SD*std, mean + SD*std
                trimmed = group.mask(group < lower, lower).mask(group > upper, upper)
                labels = pd.cut(group, [np.NINF, lower, upper, np.inf], labels=['Below','Ok','Above'])
                return pd.DataFrame({labels_name:labels, trimmed_name:trimmed})

            #filter out outliers by ppt then by items
            if ppt:
                df[['RTfiltered_by_ppt_labels','RTfiltered_by_ppt_values']] = df.groupby(['PPT','Condition','WordPos'])['RTremoveVal'].apply(outliers,'RTfiltered_by_ppt_labels','RTfiltered_by_ppt_values')

                if summarize:
                    by_ppt_filtered = df['RTfiltered_by_ppt_labels'].value_counts()
                    below = by_ppt_filtered.get("Below", 0)
                    above = by_ppt_filtered.get("Above", 0)
                    print("\nThe summary for filtering by ppt:")
                    print(">>%s or %.3f%% items were below the specified cutoff of -%s SD." % (below, below*100/total, SD))
                    print(">>%s or %.3f%% items were above the specified cutoff of +%s SD." % (above, above*100/total, SD))
                            
            if items:
                df[['RTfiltered_by_item_labels','RTfiltered_by_item_values']] = df.groupby(['Item','Condition','WordPos'])['RTremoveVal'].apply(outliers,'RTfiltered_by_item_labels','RTfiltered_by_item_values')

                if summarize:
                    by_item_filtered = df['RTfiltered_by_item_labels'].value_counts()
                    below = by_item_filtered.get("Below", 0)
                    above = by_item_filtered.get("Above", 0)
                    print("\nThe summary for filtering by items:")
                    print(">>%s or %.3f%% items were below the specified cutoff of -%s SD." % (below, below*100/total, SD))
                    print(">>%s or %.3f%% items were above the specified cutoff of +%s SD." % (above, above*100/total, SD))

            return df
        
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("There is no data loaded. Load data from a valid source before proceeding.")
            
        required_columns = ['PPT', 'WordPos', 'Condition', 'Item', 'RT']       

        missing = []
        for column in required_columns:
            if column not in self.data.columns:
                missing.append(col)
                
        if missing:
            raise ValueError('The following columns are missing in loaded data: %s' % (", ",join(missing)))

        df = self.data[pd.to_numeric(self.data['RT'], errors='coerce').notnull()]

        if isinstance(critical_conditions, list):
            if any(not isinstance(condition, str) for condition in critical_conditions):
                raise TypeError("All conditions in provided critical conditions must be of type string.")
            if any(condition not in df['Condition'].unique() for condition in critical_conditions):
                raise ValueError("All conditions provided in critical_conditions must be in supplied self.data")
        else:
            raise TypeError("Provided critical_conditions object must be of type list.")

        if summarize:
            print("\nThere are %s ppts in this file." % len(df['PPT'].unique()))
            
            conds = df['Condition'].unique()

            if len(conds) > 1:
                print("\nThe following conditions are in this file: %s" % ", ".join(conds))
            elif len(conds) == 1:
                print("\nThere is only one condition in this file: %s" % conds)
            else:
                raise ValueError("no conds loaded")

            print("\nThere are %s records." % len(df.index))

        if isinstance(remove_extremes, list):
            if len(remove_extremes) == 2:
                low, high = remove_extremes[0], remove_extremes[1]
                if low < high:
                    df = _remove_extremes(df, low, high)
                else:
                    raise ValueError("Ensure the provided low value (%s) is lower than the provided high value (%s)." % (low, high))
        elif remove_extremes != None:
            raise TypeError("Provide a list [lower, upper] or None.  Provided remove_extremes of type: %s is invalid." % type(remove_extremes))

        if identify_missing_data:
            df = _identify_missing_data(df)

        if isinstance(filter_outliers, int) or isinstance(filter_outliers, float):
            if filter_outliers < 0:
                 raise ValueError("SD provided to filter_outliers must be a positive int or float")
            elif filter_outliers < 1 or filter_outliers > 3:
                 print("NOTE: The provided SD: %s seems strange. Ensure you are providing the number of SD within which you wish to retain data." % filter_outliers)
            df = _filter_outliers(df, SD = filter_outliers)
        elif filter_outliers != None:
            raise TypeError("Provide an int or float specifying the standard deviations used for cutoff of outliers or use None to skip this step. Provided filter_outliers of type: %s is invalid." % type(filter_outliers))
        
        self.RTdata = df.reset_index()
        print("\n\nSuccessfully filtered RT data. Find the filtered data in self.RTdata")

    def get_CompQdata(self):
        """
        columns = {'PPT':'Subject', 'Condition':'condition', 'List':'', Item':'item', 'CompQAcc':'comprQ_ACC', 'CompQRT':'comprQ_RT'}
        """
        required_columns = ['PPT','Condition', 'Item','CompQAcc','CompQRT']
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("There is no data loaded. Load data from a valid source before proceeding.")
            
        
        missing = []
        for column in required_columns:
            if column not in self.data.columns:
                missing.append(column)
                
        if missing:
            raise ValueError('The following columns are missing in loaded data: %s' % (", ".join(missing)))
  
        self.CompQdata = self.data.groupby(['PPT','Condition','Item']).mean().loc[:,['CompQAcc','CompQRT']]

    def plot_reading_times(self, title, by, config, **kwargs):
        config = self.plot_configs.get(config, config)
        if isinstance(config, plot_config):
            self._plot_reading_times(by = by, conds = config.conds, words = config.words, title = title, c= config.c, fmt = config.fmt, **kwargs)

    def _plot_reading_times(self, by, conds, words, title, c = [], fmt = [], ppts = [], items = [], adj_factor = .05, e_cap = 2, e_width = 1, e_c = 'black', lw = 1, mk = 5, X = 14, Y = 7):
        if not isinstance(self.RTdata, pd.DataFrame):
            raise ValueError("Ensure that data has been loaded and filtered.")

        if by == "PPT" or by == "Item":
            print("Plotting data by %s..." % by)
        else:
            raise ValueError("Provided by: %s is invalid. Please provide 'PPT' or 'item'" % by)
        
        if not isinstance(conds, list):
            raise TypeError("Provided conds of type: %s is invalid. Provide a list of conditions." % type(conds))
        
        if isinstance(words, dict):
            if any(word not in self.RTdata['WordPos'] for word in words.keys()):
                raise ValueError('One of the provided word keys is not in self.RTdata.')
        else:
            raise TypeError('The provided words of type: %s is invalid. Provide a dict where keys correspond to wordpos and values to the corresponding word.')
        
        if isinstance(c, list):
            default_colours = ['red','maroon','yellow','olive','lime','green','aqua','teal','blue']
            if any(not isinstance(colour, str) for colour in c):
                raise TypeError("Ensure that all colours in c are strings referring to matplotlib or html colours.")
            i = 0
            while len(c) < len(conds):
                if i >= len(default_colours):
                    raise ValueError("Too many conds and not enough colours were provided. Either provide more colours or fewer conds.")
                c.append(default_colours[i])
                i += 1
        else:
            raise TypeError("Provided c of type: %s is invalid. provide a list of colours" % type(c))
        
        if isinstance(fmt, list):
            if any(not isinstance(f, str) for f in fmt):
                raise TypeError("Ensure fmt is a list of valid strings referring to matplotlib marker styles")
            while len(fmt) < len(conds):
                fmt.append(".")
        else:
            raise TypeError("Provided fmt of type: %s is invalid. Provide a list of matplotlib marker styles" % type(fmt))

        for cond in conds:
            if cond not in self.RTdata["Condition"].unique():
                raise ValueError('Could not find the provided condition: %s' % cond)
        
        conds_mask = self.RTdata["Condition"].isin(conds)
        ppts_mask = self.RTdata["PPT"].isin(ppts) if ppts else ~self.RTdata["PPT"].isnull()
        items_mask = self.RTdata["Item"].isin(items) if items else ~self.RTdata["Item"].isnull()    
        wordpos_mask = self.RTdata["WordPos"].isin(words.keys())

        mask = conds_mask & ppts_mask & items_mask & wordpos_mask
        RTcolumn = {"PPT":"RTfiltered_by_ppt_values", "Item":"RTfiltered_by_item_values"}
        df = self.RTdata[mask].groupby([by, "Condition", "WordPos"])[RTcolumn[by]].mean()
                
        x = {}
        for i in range(len(conds)):
            adj = (i - (len(conds) - 1)/2) * (adj_factor)
            x[conds[i]] = []
            for j in words.keys():
                x[conds[i]].append(j + adj)
                
        y = df.groupby(["Condition", "WordPos"]).mean().unstack()
        yerr = df.groupby(["Condition", "WordPos"]).sem().unstack()
        
        fig, ax = plt.subplots(figsize = (X,Y))
        for i in range(len(conds)):
            cond = conds[i]
            ax.errorbar(x[cond], y.loc[cond],
                yerr = yerr.loc[cond],
                c = c[i], fmt = fmt[i],
                label = cond,
                ecolor = e_c,
                capsize = e_cap,
                elinewidth = e_width,
                lw = lw,
                markersize = mk
                )

        def _format_fn(tick_val, tick_pos):
            return words.get(int(tick_val), '')

        ax.xaxis.set_major_locator(FixedLocator(list(words.keys())))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_fn))
        ax.tick_params('x', labelrotation = 45)

        plt.subplots_adjust(bottom = 0.2)
        plt.xlabel('Region', weight = 'bold', size = 'x-large')
        plt.ylabel('Reading Time (ms)', weight = 'bold', size = 'x-large')

        ax.legend()

        path = "Plots" + os.sep + "%sppts" % self.N
        if not os.path.exists(path):
            os.makedirs(path)

        if not title.endswith(".pdf"):
            title += ".pdf"

        fig.savefig(os.path.join(path, title), format = "pdf")
        fig.patch.set_facecolor("white")

    def plot_CompQAcc(self, title, by, conds = [], ppts = [], items = [], c = 'blue', X = 5, Y = 5, capsize = 10, x_tick_rotation = False):
        df = self._plot_CompQdata(by, conds, ppts, items)['CompQAcc']

        y, yerr = df.mean(), df.sem() 
        print(y)
        fig, ax = plt.subplots(figsize=(X,Y))
        ax.bar(y.index, y, yerr = yerr, color = c, capsize = capsize)
        ax.set_ylim([0,1])

        if x_tick_rotation:
            ax.set_xticklabels(y.index, rotation=45, ha='right')

        path = "Plots" + os.sep + "%sppts" % self.N
        if not os.path.exists(path):
            os.makedirs(path)

        if not title.endswith(".pdf"):
            title += ".pdf"

        fig.savefig(os.path.join(path, title), format = "pdf")

        fig.patch.set_facecolor('white')

    def plot_CompQRT(self, title, by, conds = [], ppts = [], items = [], c = 'blue', y_axis_range = None, X = 5, Y = 5, capsize = 10, x_tick_rotation = False):
        df = self._plot_CompQdata(by, conds, ppts, items)['CompQRT']

        y, yerr = df.mean(), df.sem()

        fig, ax = plt.subplots(figsize=(X,Y))
        ax.bar(y.index, y, yerr = yerr, color = c, capsize = capsize) 
        if isinstance(y_axis_range, list):
            if len(y_axis_range) == 2:
                if y_axis_range[0] < y_axis_range[1]:
                    ax.set_ylim(y_axis_range)

        if x_tick_rotation:
            ax.set_xticklabels(y.index, rotation=45, ha='right')

        path = "Plots" + os.sep + "%sppts" % self.N
        if not os.path.exists(path):
            os.makedirs(path)

        if not title.endswith(".pdf"):
            title += ".pdf"

        fig.savefig(os.path.join(path, title), format = "pdf")

        fig.patch.set_facecolor('white')

    def _plot_CompQdata(self, by, conds, ppts, items):
        if not isinstance(self.CompQdata, pd.DataFrame):
            raise ValueError("No data has been loaded in self.CompQdata.  Please load the CompQdata using self.get_CompQdata")

        idx = eval("pd.IndexSlice[%s,%s,%s]" % (ppts if ppts else ':', conds if conds else ':', items if items else ':'))
        return self.CompQdata.loc[idx, :].groupby([by,'Condition']).mean().groupby(['Condition'])
        
    def load_pickle(name):
        """
        If the Project has been generated and saved before, the pickle file can be loaded using this function returning a Project object. Loading from the pickle is faster than loading from the EMSE files each time.

        Required arguments:
        name (str) -- A string referring to the file being loaded if the pickle (.p) file is in the same directory as the Notebook (which is the default save location). If not then this requires a full file path. 

        """
        if not isinstance(name, str):
            raise TypeError("Invalid type: %s. Provide a string for the filename." % type(name))
        if not name.endswith(".p"):
            raise ValueError("Invalid file extension.  Name the file with extension: *.p")
        
        if os.path.isfile(name):
            print("Loading: %s" % name)
            return pickle.load(open(name,'rb'))
        else:
            raise ValueError("File with name: %s could not be found." % name)
    
    def save_pickle(self, name):
        """
        Saves the active project data as a pickle file in the current working directory. This file can be loaded using load_pickle instead of using load.

        Required arguments:
        name (str) -- sets the name of the pickle file to be saved (*.p)
        """
        if not isinstance(name, str):
            raise TypeError("Invalid type: %s. Provide a string for the filename." % type(name))
        if not name.endswith(".p"):
            raise ValueError("Invalid file extension.  Name the file with extension: *.p")
        
        if os.path.isfile(name):
            print("Overwriting existing pickle named: %s" % name)
        else:
            print("Creating new pickle named: %s" % name)
            
        pickle.dump(self, open(name, 'wb'))

    def get_conditions(self, conditions):
        if isinstance(conditions, list):
            if any(condition not in self.data['Condition'].unique() for condition in conditions):
                raise ValueError('One of the provided conditions is invalid.')
        else:
            raise TypeError("Provided conditions object of type: %s is invalid. Provide a list." % type(conditions))

        return self.data[self.data['Condition'].isin(conditions)]

    @property
    def N(self):
        return len(self.ppts)

    @property
    def ppts(self):
        return self.data['PPT'].unique()

    @ppts.setter
    def ppts(self, input_ppts):
        if isinstance(input_ppts, list):
            if any(ppt not in self.ppts for ppt in input_ppts):
                raise ValueError("One of the provided ppts is not in self.data")
            self.data = self.data[~self.data['PPT'].isin(input_ppts)]
        else:
            raise TypeError("Ensure you are providing a list of ppts.")

    @property
    def conditions(self):
        return self.data['Condition'].unique()

    @conditions.setter
    def conditions(self, input_conds):
        if isinstance(input_ppts, list):
            if any(cond not in self.conditions for cond in input_conds):
                raise ValueError("One of the provided conditions is not in self.data")
            self.data = self.data[~self.data['Condition'].isin(input_conds)]
        else:
            raise TypeError("Ensure you are providing a list of conditions.")

        

class plot_config:
    def __str__(self):
        output += "\n\nConditions:"
        for i in range(len(self.conds)):
            output += "\nCondition Name: %s\t\tColour: %s\t\tMarker Style: %s" % (self.conds[i], self.c[i], self.fmt[i])
        output += "\n\nSentence (x axis labels)\n%s" % " ".join([v for k,v in self.words.items()])
        return output

    def __init__(self, conds, words, c, fmt):
        if isinstance(conds, list):
            if any(not isinstance(cond, str) for cond in conds):
                raise TypeError("All conds provided must be strings representing condiitons.")
            self.conds = conds
        else:
            raise TypeError("conds must be a list. Not type: %s" % type(conds))

        if isinstance(words, dict):
            self.words = words
        else:
            raise TypeError("Provided words of type: %s is invalid. words must be a dict." % type(words))

        if isinstance(c, list):
            if any(not isinstance(colour, str) for colour in c):
                raise TypeError("all colours provided in c must be strings")

            if len(c) == len(conds):
                self.c = c
            else:
                raise ValueError('Provided c is of len: %s while provided conds is of len: %s.  c and conds must be of same length.' % (len(c), len(conds)))
        else:
            raise TypeError("Provided c is of type: %s. c must be of type list" % (type(c)))

        if isinstance(fmt, list):
            if any(not isinstance(f, str) for f in fmt):
                raise TypeError("all marker styles provided in fmt must be strings")

            if len(fmt) == len(conds):
                self.fmt = fmt
            else:
                raise ValueError('Provided fmt is of len: %s while provided conds is of len: %s.  fmt and conds must be of same length.' % (len(fmt), len(conds)))
        else:
            raise TypeError("Provided fmt is of type: %s. fmt must be of type list" % (type(c)))

QBehQ_S1_S2 = {2:'Every',3:'kid',4:'climbed',5:'a/that/those',6:'tree(s).', 7:'The', 8:'tree(s)', 9:'was/were', 10:'in', 11:'the', 12:'park.'}
QBehQ_S1 = {2:'Every',3:'kid',4:'climbed',5:'a/that/those',6:'tree(s).'}
QBehQ_S2 = {7:'The', 8:'tree(s)', 9:'was/were', 10:'in', 11:'the', 12:'park.'}

QBehQ4070_S1_S2 = {2:'Every',3:'jeweller',4:'appraised',5:'a/that/those',6:'diamond(s).', 7:'The', 8:'tree(s)', 9:'was/were', 10:'clear', 11:'and', 12:'flawless.'}
QBehQ4070_S1 = {2:'Every',3:'jeweller',4:'appraised',5:'a/that/those',6:'diamond(s).'}
QBehQ4070_S2 = {7:'The', 8:'diamond(s)', 9:'was/were', 10:'clear', 11:'and', 12:'flawless.'}

plot_configs = {'QBehQ': {'Context_Number_S1':plot_config(['AP','AS','CP','CS'],QBehQ_S1,['red','orange','green','blue'],['-^','-^','-s','-s']), 
'Context_Number_S2':plot_config(['AP','AS','CP','CS'],QBehQ_S2,['red','orange','green','blue'],['-^','-^','-s','-s']),
'Context_Number_S1_S2':plot_config(['AP','AS','CP','CS'],QBehQ_S1_S2,['red','orange','green','blue'],['-^','-^','-s','-s']),
'Context_S1':plot_config(['Control','Ambiguous'],QBehQ_S1,['blue','red'],['-s','-^']),
'Context_S2':plot_config(['Control','Ambiguous'],QBehQ_S2,['blue','red'],['-s','-^']),
'Context_S1_S2':plot_config(['Control','Ambiguous'],QBehQ_S1_S2,['blue','red'],['-s','-^']),
'Number_S1':plot_config(['Singular','Plural'],QBehQ_S1,['orange','purple'],['-s','-^']),
'Number_S2':plot_config(['Singular','Plural'],QBehQ_S2,['orange','purple'],['-s','-^']),
'Number_S1_S2':plot_config(['Singular','Plural'],QBehQ_S1_S2,['orange','purple'],['-s','-^']),
},
'QBehQ4070': {'Context_Number_S1':plot_config(['AP','AS','CP','CS'],QBehQ4070_S1,['red','orange','green','blue'],['-^','-^','-s','-s']), 
'Context_Number_S2':plot_config(['AP','AS','CP','CS'],QBehQ4070_S2,['red','orange','green','blue'],['-^','-^','-s','-s']),
'Context_Number_S1_S2':plot_config(['AP','AS','CP','CS'],QBehQ4070_S1_S2,['red','orange','green','blue'],['-^','-^','-s','-s']),
'Context_S1':plot_config(['Control','Ambiguous'],QBehQ4070_S1,['blue','red'],['-s','-^']),
'Context_S2':plot_config(['Control','Ambiguous'],QBehQ4070_S2,['blue','red'],['-s','-^']),
'Context_S1_S2':plot_config(['Control','Ambiguous'],QBehQ4070_S1_S2,['blue','red'],['-s','-^']),
'Number_S1':plot_config(['Singular','Plural'],QBehQ4070_S1,['orange','purple'],['-s','-^']),
'Number_S2':plot_config(['Singular','Plural'],QBehQ4070_S2,['orange','purple'],['-s','-^']),
'Number_S1_S2':plot_config(['Singular','Plural'],QBehQ4070_S1_S2,['orange','purple'],['-s','-^']),
}
}
