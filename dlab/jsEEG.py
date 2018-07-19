import numpy as np
import pandas as pd
import os
import re
import pickle
import matplotlib.pyplot as plt
from math import ceil

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
    def __str__(self):
        return self.summary()
       
    def summary(self, sections = ['Time', 'Electrode', 'Plotting', 'Data', 'Grands', 'Mean Amps']):
        sections_str = ", ".join(sections)
        dashes = ceil((len(sections_str) - 7)/2) + 1
        out = ["".join(["-" * dashes, "Summary", "-" * dashes])]
        out.append(sections_str)
        out.append("-" * len(out[0]))
        valid_sections = ['Time', 'Electrode', 'Plotting', 'Data', 'Grands', 'Mean Amps']
        if not all(section in valid_sections for section in sections):
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
                out.append(">\t",", ".join([grands for grands in self.grands]))
        
        if "Mean Amps" in sections:
            out.append(box("Mean Amps"))
            if len(self.mean_amps) == 0:
                out.append("No mean_amps computed.\n")
            else:
                singular = len(self.mean_amps) == 1
                out.append("There %s %s mean_amp df%s computed." % ("is" if singular else "are",
                                                        len(self.mean_amps),
                                                        "" if singular else "s"))
                out.append(">\t",", ".join([mean_amps for mean_amps in self.mean_amps]))
        
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
        The load function allows all bin files in an EMSE workspace to be loaded into the Project.data
        For this to work, call the load function and provide the path to the folder containing all ppt files (usually EMSE > Orginals)
        
        Note that this function uses the loaded settings.t and settings.electrodes to label the imported data.
        The file name is used to define the PPT # and the Condition ID. For this to work, the ppt ID must follow the last underscore in the project name.
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
                        _df = pd.DataFrame(np.fromfile(fid, np.float32).reshape((-1, len(self.settings.electrodes))), columns = self.settings.electrodes)
                        _df *= 1000000
                        _df['t'],  _df['Condition'] = self.settings.t, CID
                        _df['PPT'] = int(re.findall(r'\d+', SID)[0])
                        df = df.append(_df.set_index(['PPT','Condition']))
        self.data = df    
        
    def load_pickle(name):
        if not isinstance(name, str):
            raise TypeError("Invalid type: %s. Provide a string for the filename." % type(name))
        if not name.endswith(".p"):
            raise ValueError("Invalid file extension.  Name the file with extension: *.p")
        
        if os.path.isfile(name):
            print("Leading: %s" % name)
            return pickle.load(open(name,'rb'))
        else:
            raise ValueError("File with name: %s could not be found." % name)
    
    def save_pickle(self, name):
        if not isinstance(name, str):
            raise TypeError("Invalid type: %s. Provide a string for the filename." % type(name))
        if not name.endswith(".p"):
            raise ValueError("Invalid file extension.  Name the file with extension: *.p")
        
        if os.path.isfile(name):
            print("Overwriting existing pickle named: %s" % name)
        else:
            print("Creating new pickle named: %s" % name)
            
        pickle.dump(self, open(name, 'wb'))
        
    def compute_diffs(self):
        pass
    
    def compute_grands(self, name, ppts = []):
        if ppts:
            df = self.data.loc[ppts]
        else:
            df = self.data
            
        self.grands[name] = df.groupby(['Condition','t']).mean()
    
    def compute_mean_amps(self, name, time_windows):
        if type(time_windows) == dict:
            labels, time_windows = time_windows.keys(), time_windows.values()
        elif type(time_windows) == list:
            labels = ["t%s" % (t + 1) for t in range(len(time_windows))]
        else:
            raise TypeError("Provide a list or dict to define time_windows. Provided time_windows: %s of type %s is not valid" % (time_windows, type(time_windows)))
            
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
    
    def ppt(self, ppt_id):
        if ppt_id in self.data.index:
            return self.data.loc[ppt_id]
        else:
            raise ValueError("Provided PPT ID: %s, is not in loaded data." % (ppt_id))
        
    def plot_EEG(self, source, conditions, colours = None, electrodes='midlines', linestyles = None, fig_title = 'placeholder_title', y_axis_range = None, see_log = True, axis_formatting = True, Y = 13, X = 7):
        """
        This function allows you to plot ERPs in multiple different ways, assuming you have already loaded into conditions the necessary data.

        You may plot any number of conditions (Note that any more than 3-4 becomes very hard to read).  
            To load conditions, simply specifiy the conditions in an array: ['Cond1','Cond2']
            These conditions must already be loaded using load_data
            You may print(conditions) to see which conditions have already been loaded.

        You may also customize the colours and linestyles using arrays:
            ex: colours = ['blue','red'], linestyles = ['-',':']
            for a blue solid line and a red dashed line

        You may plot any combination of electrodes as a single plot, a row, column or grid.  
            To do so, specify an array or matrix of electrodes:
            ex: [[''F3,'Fz','F4'],['FC3','FCz','FC4'],['C3','Cz','C4']]
            Or select a preset: midlines, ROI_lateral, ROI_medial, etc

        Specify a fig_title or the file will be saved as placeholder_title.pdf

        Specify a y_axis_range as an array of 2 values: [min, max]

        Set see_log to false if you don't want the log to be printed to console (it will still save a text file)
        """
        if colours == None:
            colours = self.settings.default_colours
        
        if linestyles == None:
            linestyles = self.settings.default_linestyles
            
        electrodes = self.settings.electrode_layouts.get(electrodes, electrodes)
        
        x,y = self.dimensions(electrodes)
        fig,axes = plt.subplots(x,y,figsize=(y*Y,x*X))
        
        if isinstance(conditions, str):
            conditions = [conditions]
        elif isinstance(conditions,list):
            if not all(isinstance(condition,str) for condition in conditions):
                raise TypeError("One of the provided conditions is not a string.")
        else:
            raise TypeError("Provided conditions is of invalid type: %s. Provide a string or a list of strings" % (type(conditions)))
        
        if not all(condition in source.index for condition in conditions):
            raise ValueError("One of the provided conditions is not in the provided source.")
            
        while len(colours) < len(conditions):
            colours.append('black')
            
        while len(linestyles) < len(conditions):
            linestyles.append('-')



        def _plot_EEG(electrode,r,xaxis=True,yaxis=True):
            if electrode == None:
                r.axis('off')
            else:
                for k in range(len(conditions)):
                    linestyle = linestyles[k]
                    if(linestyle == '--'):                        
                        r.plot(self.settings.t, source.loc[conditions[k]][electrode], color=colours[k], linestyle=linestyle, dashes = (1,2))
                    else:
                        r.plot(self.settings.t, source.loc[conditions[k]][electrode], color=colours[k], linestyle=linestyle)
                        
                while isinstance(electrode,list):
                    electrode = electrode[0]
                r.text(0.025,0.9,electrode,transform=r.transAxes,fontsize=self.settings.F_size, fontweight=self.settings.F_weight)
                
                if isinstance(y_axis_range,list):
                    r.set_ylim(y_axis_range)
                    
                r.spines['bottom'].set_position('zero')
                r.spines['top'].set_color('none')
                r.spines['right'].set_color('none')
                    
                if axis_formatting:
                    if xaxis:
                        for item in r.get_xticklabels():
                            item.set_fontsize(self.settings.F_size*0.65)
                    else:
                        r.xaxis.set_ticklabels([])
                        r.tick_params(axis = 'x', length = 8)

                    if yaxis:
                        for item in r.get_yticklabels():
                            item.set_fontsize(self.settings.F_size*0.65)
                    else:
                        r.yaxis.set_ticklabels([])
                        r.tick_params(axis = 'y', length = 8)

        i = 0
        if x > 1: ####grid or col layout
            for row in axes:
                if y > 1: #####grid layout
                    j = 0
                    for electrode in electrodes[i]:
                        r = row[j]
                        _plot_EEG(electrode, r, xaxis = (i == x - 1), yaxis = (j == 0))
                        j += 1
                else: #####col layout
                    _plot_EEG(electrodes[i], row, xaxis = (i == x - 1))                
                i+=1
        elif y > 1: ####row layout
            for electrode in electrodes:
                _plot_EEG(electrode, axes[i], yaxis = (i == 0))
                i+=1
        else: ####single plot
            _plot_EEG(electrodes, axes)

        fig.set_tight_layout(True)

        path = 'Plots' + os.sep
        if not os.path.exists(path):
            os.makedirs(path)

        def _print_log():
            f = open(os.path.join(path, fig_title + '.txt'), 'w')
            f.write('Log file for the plot: ' + fig_title)
            f.write('\n')
            for i in range(len(conditions)):
                f.write('\n' + "Condition: %s\t\t--->\tColour: %s\tLinestyle: '%s'" % (conditions[i],colours[i],linestyles[i]))
            f.write('\n \n')
            f.write(str(electrodes))
            f.close()

        _print_log()
        if see_log:
            f = open(os.path.join(path, fig_title + '.txt'), 'r')
            for line in f.read().splitlines():
                print(line)
            f.close()

        fig.savefig(path + fig_title + '.pdf',format='pdf',dpi=1200)
        fig.patch.set_facecolor('white')
        print('\nPlotted successfully! Navigate to %s to find %s.pdf\n' % (path, fig_title))

    def dimensions(self, _input):
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
    
    @property
    def ppts(self):
        return self.data.index.levels[0]
    
    @property
    def conditions(self):
        return self.data.index.levels[1]
    
    @property
    def N(self):
        return len(self.ppts)
