from desktop_file_dialogs import Desktop_FilesDialog, FileGroup

import os #used to interact with PC to import and save files
import pandas as pd #pandas is the dataframe library that allows loading EEG data and various operations
import numpy as np #numpy allows various manipulations of data structures to help in plotting, constructing arrays, etc

#matplotlib import statements - this is the plotting library
import matplotlib.pyplot as plt #plotting
import matplotlib.tri as tri #tri interpolation for the topomaps
import matplotlib.patches as patches #used for drawing mask and the ears
import matplotlib.lines as lines #used for drawing ears

dfs = {}
conditions = []

electrodes = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7',
            'FC5','FC3','FC1','C1','C3','C5','T7','TP7',
            'CP5','CP3','CP1','P1','P3','P5','P7','P9',
            'PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
            'Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4',
            'F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
            'C2','C4','C6','T8','TP8','CP6','CP4','CP2',
            'P2','P4','P6','P8','P10','PO8','PO4','O2',
            'M1','M2','IO2','IO1','SO2','SO1','LO1','LO2']

sampling_rate = 1.953125 #this is 512Hz in ms ---> 1000/512
epoch = [-200,1201]
t = np.arange(epoch[0],epoch[1],sampling_rate)

#Coordinate operations for plotting topomaps
_x = np.array([-0.30882875, -0.587427189, -0.406246747, -0.286965299, -0.545007446, -0.728993475, -0.808524163, -0.950477158,
                -0.887887748, -0.676377097, -0.374709505, -0.390731128, -0.7193398, -0.933580426, -0.999390827, -0.950477158, 
                -0.887887748, -0.676377097, -0.374709505, -0.286965299, -0.545007446, -0.728993475, -0.808524163, -0.733218402, 
                -0.587427189, -0.406246747, -0.30882875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30882875, 0.587427189, 0.406246747, 
                0.0, 0.0, 0.286965299, 0.545007446, 0.728993475, 0.808524163, 0.950477158, 0.887887748, 0.676377097, 0.374709505, 
                0.0, 0.0, 0.390731128, 0.7193398, 0.933580426, 0.999390827, 0.950477158, 0.887887748, 0.676377097, 0.374709505, 
                0.286965299, 0.545007446, 0.728993475, 0.808524163, 0.733218402, 0.587427189, 0.406246747, 0.30882875])

_y = np.array([0.95047716, 0.80852416, 0.87119896, 0.71026404, 0.67302814, 0.63370436, 0.58742719, 0.30882875, 0.34082817, 
                0.35963608, 0.3747095, -0.0, -0.0, -0.0, -0.0, -0.30882875, -0.34082817, -0.35963608, -0.3747095, -0.71026404, 
                -0.67302814, -0.63370436, -0.58742719, -0.53271435, -0.80852416, -0.87119896, -0.95047716, -0.90630779, -0.99939083, 
                -0.93358043, -0.7193398, -0.39073113, 0.99939083, 0.95047716, 0.80852416, 0.87119896, 0.93358043, 0.7193398, 
                0.71026404, 0.67302814, 0.63370436, 0.58742719, 0.30882875, 0.34082817, 0.35963608, 0.3747095, 0.39073113, 
                -0.0, -0.0, -0.0, -0.0, -0.0, -0.30882875, -0.34082817, -0.35963608, -0.3747095, -0.71026404, -0.67302814, -0.63370436, 
                -0.58742719, -0.53271435, -0.80852416, -0.87119896, -0.95047716])

_z = np.array([-0.034899497, -0.034899497, 0.275637356, 0.64278761, 0.5, 0.258819045, -0.034899497, -0.034899497, 0.309016994, 
            0.64278761, 0.848048096, 0.920504853, 0.69465837, 0.35836795, -0.034899497, -0.034899497, 0.309016994, 
            0.64278761, 0.848048096, 0.64278761, 0.5, 0.258819045, -0.034899497, -0.422618262, -0.034899497, 0.275637356,
            -0.034899497, -0.422618262, -0.034899497, 0.35836795, 0.69465837, 0.920504853, -0.034899497, -0.034899497, 
            -0.034899497, 0.275637356, 0.35836795, 0.69465837, 0.64278761, 0.5, 0.258819045, -0.034899497, -0.034899497, 
            0.309016994, 0.64278761, 0.848048096, 0.920504853, 1.0, 0.920504853, 0.69465837, 0.35836795, -0.034899497, 
            -0.034899497, 0.309016994, 0.64278761, 0.848048096, 0.64278761, 0.5, 0.258819045, -0.034899497, -0.422618262,
            -0.034899497, 0.275637356, -0.034899497])

f=(1/(_z+1))
x = _x*f
y = _y*f

X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                   np.linspace(y.min(), y.max(), 100))

#electrode configs
midlines_and_lateral = [['F3','Fz','F4'],
                        ['FC3','FCz','FC4'],
                        ['C3','Cz','C4'],
                        ['CP3','CPz','CP4'],
                        ['P3','Pz','P4']]

ROI_medial=[['F3','FC1','Fz','FC2','F4'],
            ['C3','C1','Cz','C2','C4'],
            ['P3','PO3','Pz','PO4','P4']]

ROI_lateral=[['FT7','F5','F6','FT8'],
             ['F7','T7','T8','F8'],
             ['CP5','P7','P8','CP6'],
             ['PO7','O1','O2','PO8']]

midlines = [['Fz'],
            ['FCz'],
            ['Cz'],
            ['CPz'],
            ['Pz']]

#defaults
default_colours = ['black','red','blue','purple']
default_linestyles = ['-','-','-','-']
F_size, F_weight = 44, 'bold'


path = {'N4':r'C:\Users\selja\Brock University\Veena Dwivedi - Brain and Language Shared\N400Affect\Plotting\22ppts',
'dir':r'C:\Users\selja\OneDrive\Computer Stuff\Python\js_plot'}

def in_range(val,arr):
    arr = np.array(arr)
    if isinstance(val,list):
        for i in val:
            if arr.min() <= i <= arr.max():
                continue
            else:
                return False
        else:
            return True
    elif isinstance(val,int) or isisntance(val,float):
        if arr.min() <= val <= arr.max():
            return True
        else:
            return False
    else:
        raise TypeError('Provided value is of invalid type. Provide a list, int or float value')

def load_data(init_dir="",load_all=True):
    if os.path.exists(init_dir) and load_all:
        init_dir_encode = os.fsencode(init_dir)
        files = []
        for file in os.listdir(init_dir_encode):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"): 
                files.append(init_dir + os.sep + filename)
                continue
            else:
                continue
        if files:
            load_files(files)
            if len(files) == len(conditions):
                print("%s conditions successfully loaded." % len(conditions))
            else:
                raise ValueError('Unknown Error')
        else:
            raise ValueError("This directory contains no text files.  No conditions were loaded.")          
    else:
        if init_dir and not os.path.exists(init_dir):
            print("The provided path does not exist")
            print("Opening default directory...")
            init_dir = ""
        Desktop_FilesDialog(
          title             = "Select Files",
          initial_directory = init_dir,
          on_accept         = lambda file_paths: load_files(file_paths),
          on_cancel         = lambda:            print(">>> NO FILES SELECTED"),
          file_groups = [FileGroup.All_FileTypes],
        ).show()

def load_files(files):
    global directory_in_str
    for file in files:
        directory_in_str, filename = os.path.split(file)
        condition = os.path.splitext(filename)[0]
        conditions.append(condition)
        filename = directory_in_str + os.sep + filename
        dfs[condition] = read_data(filename)
        dfs[condition]['t'] = t

def read_data(file,electrode_labels=electrodes,index=False):
    return pd.read_csv(file, index_col=index, sep='\t', engine='python', names=electrode_labels)

def compute_average(input_dfs, output_name):
    if isinstance(input_dfs,list):
        _dfs = []
        for df in input_dfs:
            if isinstance(df, str):
                if df in dfs:
                    _dfs.append(dfs[df])
                else:
                    raise ValueError('Provided df: %s, was not found in dfs list' % df)
            else:
                raise TypeError('%s is not a string' % df)
        if isinstance(output_name,str):
            if output_name not in dfs:
                filepath = output_name + ".txt"
                dfs[output_name] = pd.concat(_dfs).groupby(level=0).mean()
                dfs[output_name].loc[:,'Fp1':'LO2'].to_csv(filepath, sep = '\t', index = False, header = False)
                conditions.append(output_name)
            else:
                print('A df by the name: %s already exists. Provide a different name.' % output_name)
        else:
            raise TypeError('%s is not a valid output name. Ensure you provide a string.' % output_name)
    else:
        raise TypeError('Ensure you provide a list of dfs')

def compute_difference(df_a, df_b, output_name):
    if isinstance(df_a,str) & isinstance(df_b,str) & isinstance(output_name,str):
        if df_a in dfs and df_b in dfs:
            if output_name not in dfs:
                dfs[output_name] = dfs[df_a] - dfs[df_b]
                dfs[output_name]['t'] = t
                filepath = directory_in_str + os.sep + output_name + '.txt'
                dfs[output_name].loc[:,'Fp1':'LO2'].to_csv(filepath, sep = '\t', index = False, header = False)
                conditions.append(output_name)
            else:
                print('A df named %s already exists.' % output_name)
        else:
            raise ValueError('Ensure the provided dfs: %s and %s are already in dfs' % (df_a, df_b))
    else:
        raise TypeError('Ensure the provided df names are strings')

def plot_topomap(conditions, time, vrange = None, fig_title='placeholder_title', show_sensors=False, show_head=True, nlevels=10, contour = True):
    r,c = dimensions(conditions)
    fig,axes = plt.subplots(r,c,figsize=(5*c,5*r))

    if isinstance(conditions,str):
        conditions = [conditions]

    z = {} #1D information
    Z = {} #interpolated data

    #parse time input and create appropriate z variables
    if isinstance(time,list):
        if len(time) == 2:
            if in_range(time,t):
                print(time)
                lower, upper = time[0], time[1]
                if lower < upper:
                    if r > 1:
                        for row in conditions:
                            for condition in row:
                                z[condition] = dfs[condition][(dfs[condition]['t'] > lower)&(dfs[condition]['t'] < upper)].loc[:,'Fp1':'O2'].mean()
                    else:
                        for condition in conditions:
                            z[condition] = dfs[condition][(dfs[condition]['t'] > lower)&(dfs[condition]['t'] < upper)].loc[:,'Fp1':'O2'].mean()
                else:
                    raise ValueError('Ensure that the range you provide is defined as [lower,upper]')
            else:
                raise ValueError('Provided array contains elements outside of time range')
        else:
            raise ValueError('Provided time range: %s, should only have 2 elements' % time)
    elif isinstance(time,float) or isinstance(time,int):
        if in_range(time,t):
            if time not in t:
                time_adj = time - ((time + 200) % 1.953125)
                print("Provided time: %s, was adjusted to: %s." % (time, time_adj))
                time = time_adj
            if r > 1:
                for row in conditions:
                    for condition in row:
                        z[condition] = dfs[condition][(dfs[condition]['t'] == time)].loc[:,'Fp1':'O2'].mean()
            else:
                for condition in conditions:
                    z[condition] = dfs[condition][(dfs[condition]['t'] == time)].loc[:,'Fp1':'O2'].mean()
        else:
            raise ValueError('Provided time: %s, is out of range' % time) 
    else:
        raise TypeError('Provided time: %s of %s type, is invalid. Enter a range or a single time point' % (time,type(time)))

    #set vmax and vmin from provided parameters or data
    if isinstance(vrange,list):
        if len(vrange) == 2:
            vmin, vmax = vrange[0], vrange[1]
            if vmin < vmax:
                proceed = True
            else:
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

    triangles = tri.Triangulation(x, y)
    for condition in z:
        tri_interp = tri.CubicTriInterpolator(triangles, z[condition])
        Z[condition] = tri_interp(X,Y)

    def _plot_topomap(ax, contour, Z):
        global cm

        if contour:
            cm = ax.contourf(X,Y,Z,np.arange(vmin,vmax + .1,(vmax-vmin)/nlevels),cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
        else:
            cm = ax.pcolor(X,Y,Z,cmap=plt.cm.jet, vmax=vmax, vmin=vmin)

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
                _plot_topomap(row, contour, Z[conditions[i]])
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
    path = directory_in_str + os.sep + 'Plots' + os.sep
    fig.savefig(path + fig_title + '.pdf',format='pdf')
    print('Plotted successfully! Navigate to %s to find %s.pdf' % (path, fig_title))

def plot_EEG(conditions, colours = default_colours, electrodes=midlines, linestyles = default_linestyles, fig_title = 'placeholder_title', y_axis_range = None, axis_formatting = True, axvlines = False, electrode_labels = True, Y = 13, X = 7):
    x,y = dimensions(electrodes)
    fig,axes = plt.subplots(x,y,figsize=(y*Y,x*X))

    if isinstance(conditions,list):
        length = len(conditions)
    else:
        length = 1
    while len(colours) < length:
        colours.append('black')
    while len(linestyles) < length:
        linestyles.append('-')

        
    def _plot_EEG(electrode,r):
        num_conds = length
        if electrode == None:
            r.axis('off')
        else:
            if num_conds == 1:
                r.plot(t,dfs[conditions][electrode],color=colours[0],linestyle=linestyles[0])
            else:
                for k in range(num_conds):
                    r.plot(t,dfs[conditions[k]][electrode],color=colours[k],linestyle=linestyles[k])
            if axvlines:
                r.axvline(x=0)
                r.axvline(x=600)
            if electrode_labels:
                while isinstance(electrode,list):
                    electrode = electrode[0]
                r.text(0.025,0.9,electrode,transform=r.transAxes,fontsize=F_size, fontweight=F_weight)
            if isinstance(y_axis_range,list):
                r.set_ylim(y_axis_range)
            if axis_formatting:
                r.spines['bottom'].set_position('zero')
                r.spines['top'].set_color('none')
                r.spines['right'].set_color('none')
                
    i = 0
    if x > 1: ####grid or col layout
        for row in axes:
            if y > 1: #####grid layout
                j = 0
                for electrode in electrodes[i]:
                    r = row[j]
                    _plot_EEG(electrode, r)
                    j += 1
            else: #####col layout
                _plot_EEG(electrodes[i], row)                
            i+=1
    elif y > 1: ####row layout
        for electrode in electrodes:
            _plot_EEG(electrode, axes[i])
            i+=1
    else: ####single plot
        _plot_EEG(electrodes, axes)
    fig.set_tight_layout(True)

    filename = fig_title + '.pdf'
    path = directory_in_str + os.sep + 'Plots' + os.sep
    if not os.path.exists(path):
        os.makedirs(path)

    def _print_log():
        f = open(os.path.join(path, fig_title + '.txt'), 'w')
        f.write('Log file for the plot: ' + fig_title)
        f.write('\n')
        if length == 1:
            f.write('\n' + '%s ---> Condition: %s' % (colours[0],conditions))
        else:
            for i in range(length):
                f.write('\n' + "Condition: %s\t\t--->\tColour: %s\tLinestyle: '%s'" % (conditions[i],colours[i],linestyles[i]))
        f.write('\n \n')
        f.write(str(electrodes))
        f.close()
        
    _print_log()
    fig.savefig(path + fig_title + '.pdf',format='pdf',dpi=1200)

    print('Plotted successfully! Navigate to %s to find %s.pdf' % (path, fig_title))

def dimensions(_input):
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

def plot_legend(y_axis_range, axis_formatting = True):
    if isinstance(y_axis_range,list):
        if len(y_axis_range) == 2:
            ymin, ymax = y_axis_range[0],y_axis_range[1]
            if ymin < ymax:
                fig = plt.figure(figsize=(13,7))
                ax = fig.add_subplot(111, xlabel='Time (ms)', ylabel='Voltage (uV)')
                ax.plot(t,[0]*len(t),linewidth=0)
                ax.set_ylim(y_axis_range)
                if axis_formatting:
                    ax.spines['bottom'].set_position('zero')
                    ax.spines['top'].set_color('none')
                    ax.spines['right'].set_color('none')
                    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(30)
                path = directory_in_str + os.sep + 'Plots' + os.sep
                fig.savefig(path + 'Legend' + '.pdf',format='pdf',dpi=1200)
            else:
                raise ValueError("Ensure range is provided in form [min, max]")
        else:
            raise ValueError("Ensure the range provided only contains two values")
    else:
        raise TypeError("Ensure the provided range is a list in the form [min, max]")
