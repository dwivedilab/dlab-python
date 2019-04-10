import os
import re
import pickle
from math import ceil
import numpy as np
import pandas as pd
# matplotlib import statements - this is the plotting library
import matplotlib as mpl
import matplotlib.pyplot as plt  # plotting
import matplotlib.tri as tri  # tri interpolation for the topomaps
import matplotlib.patches as patches  # used for drawing mask and the ears
import matplotlib.lines as lines  # used for drawing ears

class settings:
    electrode_layouts = {"midlines":[['Fz'],
                                     ['FCz'],
                                     ['Cz'],
                                     ['CPz'],
                                     ['Pz']],
                         
                         "midlines_and_laterals":[['F3','Fz','F4'],
                                                  ['FC3','FCz','FC4'],
                                                  ['C3','Cz','C4'],
                                                  ['CP3','CPz','CP4'],
                                                  ['P3','Pz','P4']],
                         
                         "ROI_medial":[['F3','FC1','Fz','FC2','F4'],
                                       ['C3','C1','Cz','C2','C4'],
                                       ['P3','PO3','Pz','PO4','P4']],
                         
                         "ROI_lateral":[['FT7','F5','F6','FT8'],
                                        ['F7','T7','T8','F8'],
                                        ['CP5','P7','P8','CP6'],
                                        ['PO7','O1','O2','PO8']]
                        }
    
    time_windows = {"default":[(100,300),(300,500),(500,700),(700,900),(900,1100)]}
    
    def __init__(self, 
                 sampling_interval = 1.953125, 
                 epoch = {'start':-200, 'end':1201}, 
                 electrodes_path = None,
                 default_colours = ['black','red','blue','purple'],
                 default_linestyles = ['-','-','-','-'],
                 F_size = 44,
                 F_weight = 'bold'
                ):            
        self.sampling_interval = sampling_interval
        self.epoch = epoch
        self.default_colours = default_colours
        self.default_linestyles = default_linestyles
        self.F_size = F_size
        self.F_weight = F_weight
        
        if electrodes_path == None:
            self._electrodes_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 
                "coordinates.xyz"
            )
        else:
            self._electrodes_path = electrodes_path
        self.import_electrodes(self._electrodes_path)
        
    def import_electrodes(self, fpath):
        df = pd.read_csv(fpath, index_col=False, names = ["electrodes","_x","_y","_z"])
        f = (1/(df['_z']+1))
        self.x = np.array(df["_x"]*f)
        self.y = np.array(df["_y"]*f)
        self.x = self.x[~np.isnan(self.x)]
        self.y = self.y[~np.isnan(self.y)]
        self.X, self.Y = np.meshgrid(np.linspace(self.x.min(), self.x.max(), 100),
                                     np.linspace(self.y.min(), self.y.max(), 100))
        self.electrodes = list(df['electrodes'])
    
    @property
    def t(self):
        return np.arange(self.epoch['start'],
                         self.epoch['end'],
                         self.sampling_interval)
    
class Project:
    """
    A project class which contains EEG data.
	
	Public Methods:
    summary -- print a summary of the Project, or a subset of the summary
    load -- load EMSE bin files into the Project by providing the EMSE > Original path
    load_pickle -- load a pickle file to re-initialize a Project object
    save_pickle -- save a pickle file for later re-initializing
    compute_diffs -- compute differences between conditions across ppts
    compute_avgs -- compute averages between conditions across ppts
    compute_grands -- compute grand averages and save to the self.grands dict
    compute_mean_amps -- compute mean amplitudes and save to the self.mean_amps dict
    plot_EEG -- plot ERP waveforms for a single electrode or a row/column/grid of electrodes and save as pdf
    plot_electrodes -- plot ERP waveforms for a single electrode (use for custom plotting, does not save)
    plot_topomap -- plot topographic maps
    dimensions -- for layout of eletrodes for waveform plots, and for layout of conditions for topomaps (returns: x and y)
    plot_legend -- plots a legend using teh same format as the plot_EEG function legend
    get_conditions -- retrives data for single or multiple conditions for all participants
    ppt -- retrieves a single participants data

    Public Properties:
    ppts -- returns list of participants (get/set; type = pandas.Index)
    conditions -- returns list of condition names (get/set; type = pandas.Index)
    N -- returns number of participants (get only; type = int)
    """
    def __str__(self):
        return self.summary()
       
    def summary(self, sections = ['Time', 'Electrode', 'Plotting', 'Data', 'Grands', 'Mean Amps']):
        sections_str = ", ".join(sections)
        dashes = ceil((len(sections_str) - 7)/2) + 1
        out = ["".join(["-" * dashes, "Summary", "-" * dashes])]
        out.append(sections_str)
        out.append("-" * len(out[0]))
        valid_sections = ['Time', 'Electrode', 'Plotting', 'Data', 'Grands', 'Mean Amps']
        if any(section not in valid_sections for section in sections):
            raise ValueError("A provided section is not valid. Please provide any/all of the following: %s" % ", ".join(valid_sections))
            
        def box(title):
            line = "-" * (len(title) + 2)
            return "\n%s\n|%s|\n%s\n" % (line, title, line)
        
        if "Time" in sections:
            out.append(box("Time"))
            out.append("There are %s timepoints in each epoch." % (len(self.settings.t)))
            out.append("Epochs are %sms to %sms" % (self.settings.epoch['start'], self.settings.epoch['end']))

            singular = len(self.settings.time_windows) == 1
            out.append("There %s %s time window config%s loaded." % ("is" if singular else "are",
                                                             len(self.settings.time_windows),
                                                             "" if singular else "s"))
            for time_windows in self.settings.time_windows:
                out.append(">\t%s: %s" % (time_windows, self.settings.time_windows[time_windows]))
        
        if "Electrode" in sections:
            out.append(box("Electrode"))
            out.append("There are %s electrodes loaded: %s\n" % (len(self.settings.electrodes), ", ".join(self.settings.electrodes)))
            out.append("There are %s electrode layouts:" % len(self.settings.electrode_layouts))
            for electrodes in self.settings.electrode_layouts:
                out.append("\tLayout name: %s\n%s" % (electrodes, "".join(["\t>\t" + str(i) + "\n" for i in self.settings.electrode_layouts[electrodes]])))
        
        if "Plotting" in sections:
            out.append(box("Plotting"))
            out.append("Default colours: %s" % ", ".join(self.settings.default_colours))
            out.append("Default linestyles: %s" % ", ".join(self.settings.default_linestyles))
            out.append("Font size of electrode labels: %s pts." % self.settings.F_size)
            out.append("Font size of axis labels: %s pts." % int(self.settings.F_size * 0.65))
        
        if "Data" in sections:
            out.append(box("Data"))
            if len(self.data) == 0:
                out.append("No data loaded.\n")
            else:
                ##out.append("Data was loaded from: %s\n" % self.data_path)
                out.append("There are %s records.\n" % len(self.data))
                singular = len(self.data) == 1
                out.append("There %s %s ppt%s. \nPPTs: %s\n" % ("is" if singular else "are",
                                                           self.N,
                                                           "" if singular else "s",
                                                           ", ".join(str(ppt) for ppt in self.ppts)))
                out.append("There are %s conditions. \nConditions: %s\n" % (len(self.conditions), ", ".join(list(self.conditions))))
            
        if "Grands" in sections:
            out.append(box("Grands"))
            if len(self.grands) == 0:
                out.append("No grands computed.\n")
            else:
                singular = len(self.grands) == 1
                out.append("There %s %s grands df%s computed." % ("is" if singular else "are",
                                                        len(self.grands),
                                                        "" if singular else "s"))
                out.append(">\t" + ", ".join([grands for grands in self.grands]))
        
        if "Mean Amps" in sections:
            out.append(box("Mean Amps"))
            if len(self.mean_amps) == 0:
                out.append("No mean_amps computed.\n")
            else:
                singular = len(self.mean_amps) == 1
                out.append("There %s %s mean_amp df%s computed." % ("is" if singular else "are",
                                                        len(self.mean_amps),
                                                        "" if singular else "s"))
                out.append(">\t" + ", ".join([mean_amps for mean_amps in self.mean_amps]))
        
        return "\n".join(out)
    
    def __init__(self, my_settings = settings()):
        self.data = pd.DataFrame()
        self.grands = {}
        self.mean_amps = {}
        if isinstance(my_settings, settings):
            self.settings = my_settings
        else:
            raise TypeError("Please provide a valid settings object")

    def load(self, path):
        """
        The load function allows all bin files in an EMSE workspace to be loaded into self.data. Note that this function uses the loaded settings.t and settings.electrodes to label the imported data.  The file name is used to define the PPT # and the Condition ID. For this to work, the ppt ID must follow the last underscore in the project name.
        
        Required arguments:
        path (str) -- provide a path to where the bins are saved as a string. Ex: r'FULLPATHHERE'
        """
        self.data_path = path
        def split(fname):
            SID = fname.split("_")[-1]
            CID = fname[:-1 - len(SID)]
            return SID[:-4], CID
        
        df = pd.DataFrame()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".bin"):
                    fpath = os.path.join(root, file)
                    with open(fpath, 'rb') as fid:
                        SID, CID = split(file)
                        try:
                            _df = pd.DataFrame(np.fromfile(fid, np.float32).reshape((-1, len(self.settings.electrodes))), columns = self.settings.electrodes)
                            _df *= 1000000
                            _df['t'],  _df['Condition'] = self.settings.t, CID
                            _df['PPT'] = int(re.findall(r'\d+', SID)[0])
                            df = df.append(_df.set_index(['PPT','Condition']))
                        except:
                            print('Failed to load %s for ppt %s (file name = %s)' % (CID, SID, file))
        self.data = df
        # double check to make sure everything is right!
        print('\n%s condition(s) for %s ppt(s) with %s time slices were loaded' % (len(self.conditions), self.N, len(self.settings.t)))
        expected_records = len(self.conditions) * self.N * len(self.settings.t)
        print('So we expect to see %s records.' % (expected_records))
        print('%s records were actually loaded.' % len(self.data))
        if expected_records == len(self.data):
            print('Everything looks in order!\n')
        else:
            print("Something doesn't look right - double check to make sure.\n")
        
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
        
    def compute_diffs(self, minuend, subtrahend, difference):
        """
        Compute difference scores between conditions for each ppt and store it back into data: minuend - subtrahend = difference
        
        Required arguments:
        minuend (str) -- the condition id for the minuend
        subtrahend (str) -- the condition id for the subtrahend
        difference (str) -- the condition id that the difference will be named
        """
        minuend_df, subtrahend_df = self.get_conditions(minuend).reset_index(), self.get_conditions(subtrahend).reset_index()
        if difference in self.conditions:
            print("Note that a condition named %s already exists in this project." % difference)
        
        first_electrode, second_electrode = self.settings.electrodes[0], self.settings.electrodes[-1]
        
        difference_df =  minuend_df.loc[:,first_electrode:second_electrode] - subtrahend_df.loc[:,first_electrode:second_electrode] 
        difference_df['PPT'], difference_df['t'] = minuend_df['PPT'], minuend_df['t']
        difference_df['Condition'] = difference
        
        self.data = pd.concat([self.data, difference_df.set_index(['PPT','Condition'])], sort = False)
        
        print("Successfully computed difference named %s from: %s = %s - %s" % (difference, difference, minuend, subtrahend))
        print("This has been saved back to data.  Note that you will need to update any mean_amps or grands that have already computed.")
    
    def compute_avgs(self, inputs, output):
        """
        Compute an average of certain conditions for each participant
        
        Required arguments:
        inputs (list of str) -- a list of conditions that are in the data to be average for each ppt
        output (str) -- a condition name for the output average
        """
        if type(inputs) != list:
            raise TypeError("Provided inputs list is of type: %s. Please provide a list of strings." % type(inputs))
        
        if any(input not in self.conditions for input in inputs):
            raise ValueError("One of the provided conditions was not found.")
			
        idx = pd.IndexSlice
        output_df = self.data.loc[idx[:,inputs],:].reset_index().groupby(['PPT','t']).mean().reset_index()
        output_df['Condition'] = output
        
        self.data = pd.concat([self.data.reset_index(), output_df], sort = False).set_index(['PPT','Condition'])
        print("Successfully computed average named %s from the following conditions: %s" % (output, ", ".join(inputs)))
        print("This has been saved back to data.  Note that you will need to update any mean_amps or grands that have already computed.")
    
    def compute_grands(self, name, ppts = []):
        """
        Compute grand averages for all participants or for a subset of participants

        Required arguments:
        name (str) -- the key under which this grands DataFrame will be saved in the dict self.grands

        Optional arguments:
        ppts (list of int) -- the ppts that will be included in these grands. default: [] which includes all ppts 
        """
        if ppts:
            df = self.data.loc[ppts]
        else:
            df = self.data
            
        self.grands[name] = df.groupby(['Condition','t']).mean()
    
    def compute_mean_amps(self, name, time_windows = 'default'):
        """
        Compute mean amplitudes for specific time_windows

        Required arguments:
        name (str) -- the key under which this mean_amps DataFrame will be saved in the dict self.mean_amps

        Optional arguments:
        time_windows (str, list or dict) -- a str key for predefined time_windows in self.settings.time_windows OR a list of time windows (automatically will be named t1, t2, etc) OR a dict with time windows and custom labels. default: 'default'
        """
        if isinstance(time_windows, str):
            if time_windows in self.settings.time_windows:
                time_windows = self.settings.time_windows.get(time_windows)
            else:
                raise ValueError('Provided time_windows, if str, must be a valid key for self.settings.time_windows. Could not find provided time_windows %s' % time_windows)
        
        if type(time_windows) == dict:
            labels, time_windows = time_windows.keys(), time_windows.values()
        elif type(time_windows) == list:
            labels = ["t%s" % (t + 1) for t in range(len(time_windows))]
        else:
            raise TypeError("Provide a str key for pre-defined time_windows in self.settings.time_windows or provide a list or dict to define time_windows. Provided time_windows: %s of type %s is not valid" % (time_windows, type(time_windows)))
            
        if not all((type(time_window) == tuple) & (len(time_window) == 2) for time_window in time_windows):
            raise TypeError("Ensure that all provided time windows are tuples of 2 elements.")
        
        df = self.data
        df['time_windows'] = pd.cut(df['t'], pd.IntervalIndex.from_tuples(time_windows))
        d = dict(zip(pd.Categorical(df['time_windows']).categories,labels))
        df['time_windows'].replace(d, inplace = True)
        df = df.groupby(["PPT","Condition","time_windows"]).mean()
        
        mean = pd.melt(df.drop(columns=["t"]).reset_index(),
                       id_vars=["PPT","Condition","time_windows"],
                       var_name = "electrode",
                       value_name = "Mean Amplitude")
        mean["Label"] = mean["Condition"] + mean["electrode"] + mean["time_windows"]
        
        self.mean_amps[name] = mean
    
    def get_conditions(self, condition_id):
        """
        Retrieve a subset of conditions (single or multiple) from self.data

        Required arguments:
        condition_id (str or list) -- a condition id or a list of condition ids that are in self.data
        """
        idx = pd.IndexSlice
        if type(condition_id) == list:
            if any(condition not in self.conditions for condition in condition_id):
                raise ValueError("One of the provided conditions is not in loaded data.")
            else:
                return self.data.loc[idx[:,condition_id],:]
        else:
            if condition_id in self.conditions:
                return self.data.loc[idx[:,condition_id],:]
            else:
                raise ValueError("Provided condition: %s, is not in loaded data." % (condition_id))
    
    def ppt(self, ppt_id):
        """
        Retrieve data for a single ppt

        Required arguments:
        ppt_id (int) -- an int ppt id in self.data
        """
        if ppt_id in self.data.index.get_level_values('PPT'):
            return self.data.loc[ppt_id]
        else:
            raise ValueError("Provided PPT ID: %s, is not in loaded data." % (ppt_id))
    
    def plot_topomap(self, source, conditions, time, vrange = None, fig_title='placeholder_title', show_sensors=False, show_head=True, nlevels=10, contour = True, X = 5, Y = 5):
        """
        Plot a topomap of one or multiple conditions with either contourf or pcolor
        
        Required arguments:
        source (pandas df) -- this is a source for the data, this can be any dataframe with conditions as the index, electrodes as columns and a 't' column. It will likely be a self.grands['NAME'] OR self.ppt(PPTID)
		conditions (str or list) -- this can be a single string, a one dimensional or two dimensional list of strings. Ex: 'Condition1' OR ['Condition1','Condition2'] OR [['Condition1','Condition2'],['Condition3','Condition4']]
		time (list, int or float) -- this should be a list of 2 values [lower, upper], or a single timepoint (int or float)
        
		Optional arguments:
        vrange (list or None) -- if None, uses the min and max values of the data. If list, uses the structure [lower, upper]. default: None
		fig_title (str) -- the name the pdf will be saved as. default: 'placeholder_title'
        show_sensors (bool) -- if True, dots will be placed on the plot to represent where sensors may be found. default: False
		show_head (bool) -- if True, the head outline will be shown. default: True
		nlevels (int) -- number of levels if a contour is used.  If contour style not used, this argument is ignored. default: 10
		contour (bool) -- if True, contourf will be used with number levels specified by nlevels. Else, pcolor will be used. default: True
		X (int or float) -- This value times the number of plots on the x axis determines the length of the plot. Tinker with this value and Y if the aspect ratio is off. default: 5
		Y (int or float) -- This value times the number of plots on the y axis determines the height of the plot. Tinker with this value and X if the aspect ratio is off. default: 5
        """
        def _in_range(val,arr):
            arr = np.array(arr)
            if isinstance(val,list):
                for i in val:
                    if arr.min() <= i <= arr.max():
                        continue
                    else:
                        return False
                else:
                    return True
            elif isinstance(val,int) or isinstance(val,float):
                if arr.min() <= val <= arr.max():
                    return True
                else:
                    return False
            else:
                raise TypeError('Provided value is of invalid type. Provide a list, int or float value')

        first_electrode = self.settings.electrodes[0]
        last_electrode = self.settings.electrodes[len(self.settings.x) - 1]
        r,c = self.dimensions(conditions)
        fig,axes = plt.subplots(r,c,figsize=(X*c,Y*r))

        if isinstance(conditions,str):
            conditions = [conditions]

        z = {} #1D information
        Z = {} #interpolated data

        idx = pd.IndexSlice

        #parse time input and create appropriate z variables
        if isinstance(time,list):
            if len(time) == 2:
                if _in_range(time,self.settings.t):
                    print(time)
                    lower, upper = time[0], time[1]
                    if lower < upper:
                        if r > 1:
                            for row in conditions:
                                for condition in row:
                                    z[condition] = source.loc[idx[condition, lower:upper],first_electrode:last_electrode].mean()
                        else:
                            for condition in conditions:
                                z[condition] = source.loc[idx[condition, lower:upper],first_electrode:last_electrode].mean()
                    else:
                        raise ValueError('Ensure that the range you provide is defined as [lower,upper]')
                else:
                    raise ValueError('Provided array contains elements outside of time range')
            else:
                raise ValueError('Provided time range: %s, should only have 2 elements' % time)
        elif isinstance(time,float) or isinstance(time,int):
            if _in_range(time,self.settings.t):
                if time not in self.settings.t:
                    time_adj = time - ((time - self.settings.epoch['start']) % self.settings.sampling_interval)
                    print("Provided time: %s, was adjusted to: %s." % (time, time_adj))
                    time = time_adj
                if r > 1:
                    for row in conditions:
                        for condition in row:
                            z[condition] = source.loc[idx[condition, time],first_electrode:last_electrode]
                else:
                    for condition in conditions:
                        z[condition] = source.loc[idx[condition, time],first_electrode:last_electrode]
            else:
                raise ValueError('Provided time: %s, is out of range' % time) 
        else:
            raise TypeError('Provided time: %s of %s type, is invalid. Enter a range or a single time point' % (time,type(time)))

        #set vmax and vmin from provided parameters or data
        if isinstance(vrange,list):
            if len(vrange) == 2:
                vmin, vmax = vrange[0], vrange[1]
                if vmin > vmax:
                    raise ValueError('Ensure that vrange is provided in form [min,max].')
            else:
                raise ValueError('Provided vrange has %s elements. Should only have 2.' % (len(vrange)))
        elif vrange == None:
            vmin, vmax = np.inf, np.NINF
            for condition, data in z.items():
                temp_vmin, temp_vmax = data.min(), data.max() 
                if temp_vmax > vmax:
                    vmax = temp_vmax
                if temp_vmin < vmin:
                    vmin = temp_vmin
            vmin *= 1.1
            vmax *= 1.1
        else:
            raise TypeError('Provided vrange: %s of %s type is invalid. Enter a range [min,max] or None.' % (vrange,type(vrange)))

        print('The range is %s to %s.' % (vmin,vmax))

        triangles = tri.Triangulation(self.settings.x, self.settings.y)
        for condition in z:
            tri_interp = tri.CubicTriInterpolator(triangles, z[condition])
            Z[condition] = tri_interp(self.settings.X, self.settings.Y)

        def _plot_topomap(ax, contour, Z):
            global cm

            if contour:
                cm = ax.contourf(self.settings.X,self.settings.Y,Z,np.arange(vmin,vmax + .1,(vmax-vmin)/nlevels),cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
            else:
                cm = ax.pcolor(self.settings.X, self.settings.Y,Z, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)

            #formatting changes to set the plot size and remove axes 
            ax.axis('off')
            ax.set_ylim([-1.2,1.2])
            ax.set_xlim([-1.2,1.2])

            #mask electrodes that don't fit in the circle i.e. PO3 PO4 and Iz
            mask = patches.Wedge((0,0),1.6,0,360,width=0.6, color='white')
            ax.add_artist(mask)

            if show_sensors:
                ax.plot(x,y, color = "#444444", marker = "o", linestyle = "", markersize=2)

            if show_head:
                #draw
                head_border = plt.Circle((0, 0), 1, color='black', fill=False)
                LNose = lines.Line2D([-0.087,0],[0.996,1.1], color='black', solid_capstyle = 'round', lw = 1)
                RNose = lines.Line2D([ 0.087,0],[0.996,1.1], color='black', solid_capstyle = 'round', lw = 1)
                LEar = patches.Wedge((-1,0), 0.1, 90, 270, width=0.0025, color='black')
                REar = patches.Wedge((1,0), 0.1, 270, 90, width=0.0025, color='black')

                #add
                ax.add_artist(head_border)
                ax.add_line(LNose)
                ax.add_line(RNose)
                ax.add_artist(LEar)
                ax.add_artist(REar)

        i = 0
        if r > 1: #grid or col
            for row in axes:
                if c > 1: #grid
                    j = 0
                    for ax in row:
                        _plot_topomap(ax, contour, Z[conditions[i][j]])
                        j += 1
                else: #col
                    _plot_topomap(row, contour, Z[conditions[i][0]])
                i += 1
        elif c > 1: #row
            for ax in axes:
                _plot_topomap(ax, contour, Z[conditions[i]])
                i += 1
        else: #single plot
            _plot_topomap(axes, contour, Z[conditions[0]])

        #adjust plot and add color bar
        fig.subplots_adjust(right=0.8, top = 0.85, bottom = 0.15)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(cm, cax = cbar_ax)

        #save file
        path = os.path.join("Plots", "%sppts" % self.N, "EEG")
        self._save_fig(fig, path, fig_title)

        with open(os.path.join(path, fig_title + '.txt'), 'w') as f:
            f.write('Log file for the plot: %s\n\n' % fig_title)
            f.write('time: %s ms\n' % time)
            f.write('vrange: %s uV  to %s uV.\n\n' % (vmin,vmax))
            f.write('Conditions:\n')
            f.write(str(conditions))

        fig.patch.set_facecolor('white')
        print('Plotted successfully! Navigate to %s to find %s' % (path, fig_title))
    
    def plot_EEG(self, source, conditions, electrodes='midlines', colours = None, linestyles = None, fig_title = 'placeholder_title', y_axis_range = None, see_log = True, axis_formatting = True, Y = 13, X = 7):
        """
        Plot ERP waveforms for any number of conditions (optimal viewing at 1-4 conditions) with any colours, linestyles and arrangement of electrodes. You must set a y_axis_range or each electrode plot will have its own y_axis_range

        Required arguments:
        source (pandas df) -- this is a source for the data, this can be any dataframe with conditions as the index, electrodes as columns and a 't' column. It will likely be a self.grands['NAME'] OR self.ppt(PPTID)
        conditions (str or list) -- this can be a single string or a list of strings. Ex: 'Condition1' OR ['Condition1','Condition2']

        Optional arguments:
        electrodes (str or list of str or list of list of str) -- a string specifying a layout found in self.settings.electrode_layouts OR a 1 or 2 dimensional list of electrodes ex: 'midlines' OR ['Fz','FCz'], [['Fz'],['FCz']], [['Fz','FCz'],['Cz','CPz']] default: 'midlines'
        colours (None or list of str) -- a list of colours as strings (ex: ['black','red']). The number of colours should match the number of provided conditions or left as None for default colours = ['black', 'red', 'blue', 'purple', ... all others default to black]
        linestyles (None or list) -- a list of linestyles as strings (allowed = ':' for dotted, '-' for solid, '-.' for dash and dot, '--' for dashed) or left as None for default linestyles = all solid
        fig_title (str) -- the name the pdf will be saved as. default: 'placeholder_title'
        y_axis_range (None or list of int) -- the range of the y axis as [lower, upper] or if left as None, the range is left as default for each individual plot. default: None
        see_log (bool) -- if True, print the log text file. Regardless, the log will be printed as a text file with name fig_title. default: True

        Optional arguments you shouldn't need to change:
        axis_formatting (bool) -- if True, apply custom axis formatting. Debugging use only. default: True
        Y, X (int) -- both Y and X can be set to change the aspect ratio.  13 and 7 have been set as defaults respectively through trial and error.
        """
        if colours == None:
            colours = self.settings.default_colours
        
        if linestyles == None:
            linestyles = self.settings.default_linestyles
        
        while len(colours) < len(conditions):
            colours.append('black')
            
        while len(linestyles) < len(conditions):
            linestyles.append('-')
        
        if isinstance(conditions, str):
            conditions = [conditions]
        elif isinstance(conditions,list):
            if any(not isinstance(condition,str) for condition in conditions):
                raise TypeError("One of the provided conditions is not a string.")
        else:
            raise TypeError("Provided conditions is of invalid type: %s. Provide a string or a list of strings" % (type(conditions)))
        
        if any(condition not in source.index for condition in conditions):
            raise ValueError("One of the provided conditions is not in the provided source.")

        if isinstance(electrodes, str):
            if electrodes in self.settings.electrode_layouts:
                electrodes = self.settings.electrode_layouts.get(electrodes)
            elif electrodes in self.settings.electrodes:
                electrodes = [electrodes]
            else:
                raise ValueError('If providing a str for electrodes, electrodes must be a valid electrode found in self.settings.electrodes or a name of a valid layout found in self.settings.electrode_layouts. %s is not valid' % electrodes)
        
        x,y = self.dimensions(electrodes)
        fig,axes = plt.subplots(x,y,figsize=(y*Y,x*X))
            

        kwargs = {
            'source':source, 
            'conditions':conditions, 
            'colours':colours, 
            'linestyles':linestyles, 
            'y_axis_range':y_axis_range, 
            'axis_formatting':axis_formatting
        }
        if x > 1: ####grid or col layout
            for i, row in enumerate(axes):
                if y > 1: #####grid layout
                    for j, electrode in enumerate(electrodes[i]):
                        self.plot_electrode(ax = row[j], electrode = electrode, xaxis = (i == x - 1), yaxis = (j == 0), **kwargs)
                else: #####col layout
                    self.plot_electrode(ax = row, electrode = electrodes[i], xaxis = (i == x - 1), **kwargs)
        elif y > 1: ####row layout
            for i, electrode in enumerate(electrodes):
                self.plot_electrode(ax = axes[i], electrode = electrode, yaxis = (i == 0), **kwargs)
        else: ####single plot
            self.plot_electrode(ax = axes, electrode = electrodes, **kwargs)

        fig.set_tight_layout(True)

        path = os.path.join("Plots", "%sppts" % self.N, "EEG")
        self._save_fig(fig, path, fig_title)

        with open(os.path.join(path, fig_title + '.txt'), 'w') as f:
            f.write('Log file for the plot: ' + fig_title)
            f.write('\n')
            for i in range(len(conditions)):
                f.write('\n' + "Condition: %s\t\t--->\tColour: %s\tLinestyle: '%s'" % (conditions[i],colours[i],linestyles[i]))
            f.write('\n \n')
            f.write(str(electrodes))

        if see_log:
            with open(os.path.join(path, fig_title + '.txt'), 'r') as f:
                for line in f.read().splitlines():
                    print(line)

        fig.patch.set_facecolor('white')
        print('\nPlotted successfully! Navigate to %s to find %s\n' % (path, fig_title))
    
    def _save_fig(self, fig, path, fig_title):
        if not os.path.exists(path):
            os.makedirs(path)
            
        if isinstance(fig_title, str):
            if not fig_title.endswith(".pdf"):
                fig_title += '.pdf'
        else:
            raise TypeError("The provided filename is of type: %s. Please provide a string for the filename." % (type(filename)))

        fig.savefig(os.path.join(path, fig_title),format='pdf',dpi=1200)
    
    def plot_electrode(self, source, conditions, electrode, colours, linestyles, y_axis_range = None, ax = None, axis_formatting = True, xaxis=True, yaxis=True):
        """
        Plot ERP waveforms for any number of conditions (optimal viewing at 1-4 conditions) with any colours, linestyles for a single electrode. This is intended for use in a custom plotting layout. If you wish to plot a single electrode, use plot_EEG instead.

        Required arguments:
        source (pandas df) -- this is a source for the data, this can be any dataframe with conditions as the index, electrodes as columns and a 't' column. It will likely be a self.grands['NAME'] OR self.ppt(PPTID)
        conditions (str or list) -- this can be a single string or a list of strings. Ex: 'Condition1' OR ['Condition1','Condition2']

        Optional arguments:
        electrodes (str or list of str or list of list of str) -- a string specifying a layout found in self.settings.electrode_layouts OR a 1 or 2 dimensional list of electrodes ex: 'midlines' OR ['Fz','FCz'], [['Fz'],['FCz']], [['Fz','FCz'],['Cz','CPz']] default: 'midlines'
        colours (None or list of str) -- a list of colours as strings (ex: ['black','red']). The number of colours should match the number of provided conditions or left as None for default colours = ['black', 'red', 'blue', 'purple', ... all others default to black]
        linestyles (None or list) -- a list of linestyles as strings (allowed = ':' for dotted, '-' for solid, '-.' for dash and dot, '--' for dashed) or left as None for default linestyles = all solid
        y_axis_range (None or list of int) -- the range of the y axis as [lower, upper] or if left as None, the range is left as default. default: None

        Optional arguments you shouldn't need to change:
        axis_formatting (bool) -- if True, apply custom axis formatting. Debugging use only. default: True
        """
        if colours == None:
            colours = self.settings.default_colours
        
        if linestyles == None:
            linestyles = self.settings.default_linestyles
        
        while len(colours) < len(conditions):
            colours.append('black')
            
        while len(linestyles) < len(conditions):
            linestyles.append('-')
        
        if isinstance(conditions, str):
            conditions = [conditions]
        elif isinstance(conditions,list):
            if any(not isinstance(condition,str) for condition in conditions):
                raise TypeError("One of the provided conditions is not a string.")
        else:
            raise TypeError("Provided conditions is of invalid type: %s. Provide a string or a list of strings" % (type(conditions)))
        
        if any(condition not in source.index for condition in conditions):
            raise ValueError("One of the provided conditions is not in the provided source.")

        if ax == None:
            fig, ax = plt.subplots(1)
        else:
            if isinstance(ax, mpl.axes.Axes):
                fig = ax.get_figure()
            else:
                raise TypeError('Provided ax must be a valid matplotlib axes object.')

        if electrode == None:
            ax.axis('off')
        else:
            for cond, col, linestyle in zip(conditions, colours, linestyles):
                if(linestyle == '--'):                        
                    ax.plot(self.settings.t, source.loc[cond][electrode], color=col, linestyle=linestyle, dashes = (1,2))
                else:
                    ax.plot(self.settings.t, source.loc[cond][electrode], color=col, linestyle=linestyle)
                    
            while isinstance(electrode,list):
                electrode = electrode[0]
            ax.text(0.025,0.9,electrode,transform=ax.transAxes,fontsize=self.settings.F_size, fontweight=self.settings.F_weight)
            
            if isinstance(y_axis_range,list):
                ax.set_ylim(y_axis_range)
                
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
                
            if axis_formatting:
                if xaxis:
                    for item in ax.get_xticklabels():
                        item.set_fontsize(self.settings.F_size*0.65)
                else:
                    ax.xaxis.set_ticklabels([])
                    ax.tick_params(axis = 'x', length = 8)

                if yaxis:
                    for item in ax.get_yticklabels():
                        item.set_fontsize(self.settings.F_size*0.65)
                else:
                    ax.yaxis.set_ticklabels([])
                    ax.tick_params(axis = 'y', length = 8)
        
        return fig, ax

    def dimensions(self, _input):
        """
        Used to calculate the dimensions of electrodes in plot_EEG and conditions in plot_topomap. Is left as a public function in the event the user wants to test out a list before using in a function.

        Required arguments:
        _input (str, list of str or list of list of str)
        """
        if isinstance(_input,list):
            if isinstance(_input[0],list):
                x = len(_input)
                y = len(_input[0])
                for row in _input:
                    if len(row) != y:
                        raise ValueError('Electrode matrix is not uniform, please ensure you have inserted an equal number of items in each row.  For plotting ERPs, you may type "None" where appropriate')
            else:
                x = 1
                y = len(_input)
        else:
            x,y = 1,1
        return x,y
    
    def plot_legend(self, y_axis_range, ax = None, axis_formatting = True, fontsize = None):
        """
        Create a legend in the Plots folder with specified y_axis_range

        Required arguments:
        y_axis_range (list of int or float) -- a list specified as [lower, upper]

        Optional arguments:
        ax (matplotlib.axes.Axes)
        fontsize (int)

        Optional arguments you should not need to change:
        axis_formatting (bool) -- if True, apply custom axis formatting. Debugging use only. default: True
        """
        if fontsize is None:
            fontsize = self.settings.F_size

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(13,7))
        elif isinstance(ax, mpl.axes.Axes):
            fig = ax.get_figure()
        else:
            raise TypeError('Provided ax must be a valid matplotlib axes object.')


        if not isinstance(y_axis_range, list):
            raise TypeError("Ensure the provided range is a list in the form [min, max]")

        if len(y_axis_range) == 2:
            ymin, ymax = y_axis_range[0],y_axis_range[1]
        else:
            raise ValueError("Ensure the range provided only contains two values")

        if ymin >= ymax:
            raise ValueError("Ensure range is provided in form [min, max]")

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (uV)')
        ax.plot(self.settings.t,[0]*len(self.settings.t),linewidth=0)
        ax.set_ylim(y_axis_range)
        if axis_formatting:
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

        return ax, fig

    @property
    def ppts(self):
        """
        Returns the list of all participants loaded in self.data.
        """
        return self.data.index.get_level_values('PPT').unique()
    
    @ppts.setter
    def ppts(self, input_ppts):
        """
        Sets the participants to be used from EEG.data
        
        Required Arguments:
        input_ppts (list of int) -- list of participant numbers to be used
        """
        if not isinstance(input_ppts,list):
            raise TypeError("input_ppts must be of type list")

        if any(ppt not in self.ppts for ppt in input_ppts):
            raise ValueError("Ensure all provided ppts are in self.data")

        idx = pd.IndexSlice
        self.data = self.data.loc[idx[input_ppts,:],:].reset_index().set_index(['PPT','Condition'])

    @property
    def conditions(self):
        """
        Returns the list of all conditions loaded in self.data
        """
        return self.data.index.get_level_values('Condition').unique()
    
    @conditions.setter
    def conditions(self, input_conditions):
        """
        Sets the conditions to be used from EEG.data
        
        Required Arguments:
        input_conditions (list of str) -- list of conditions to be used
        """
        if not isinstance(input_conditions,list):
            raise TypeError("input_ppts must be of type list")

        if any(condition not in self.conditions for condition in input_conditions):
            raise ValueError("Ensure all provided conditions are in self.data")
            
        idx = pd.IndexSlice
        self.data = self.data.loc[idx[:,input_conditions],:].reset_index().set_index(['PPT','Condition'])

    @property
    def N(self):
        """
        Returns the number of participants loaded in self.data
        """
        return len(self.ppts)
