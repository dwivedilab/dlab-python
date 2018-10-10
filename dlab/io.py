import os
import pandas as pd
from savReaderWriter import * 

def import_from_excel(filename, sheetname):
    """
    Imports data from an Excel FIle (.xlsx) into a pandas.DataFrame called 'xl'. 

    Required Arguments:
    filename (str) -- file name of .xlsx file to be imported as pandas.DataFrame.
    sheetname (str) -- name of the worksheet in the .xlsx file to be imported as pandas.DataFrame.
    """
    if os.path.isfile(filename):
        xl = pd.ExcelFile(filename)
    else:
        raise ValueError('Provided file "%s" cannot be found' % filename)
    
    if sheetname in xl.sheet_names:
        return xl.parse(sheetname)
    else:
        raise ValueError('Provided sheet "%s" cannot be found' % sheetname)

def export_to_excel(filename, dfs, output_sheet_names):
    """
    Exports and saves the supplied pandas.DataFrame(s) to separate sheets in a single excel file (.xlsx).  This should be run at the end of a notebook.
    
    Required Arguments:
    filename (str) -- file name of exported excel file. Must end with '.xlsx'
    dfs (list of pandas.DataFrame) -- pandas.DataFrame(s) to be exported to Excel provided as a [list]. To write multiple sheets in one Excel File, provide each DataFrame in the [list]. 
    output_sheet_names (list of str) -- Name of the sheet in the exported Excel file. For each DataFrame provided in 'dfs', a sheet name must also be supplied as a [list].
    """
    print("If you are suppling a '.groupby' pandas object as a 'dfs', it is recommended that you use '.unstack' method on the object for this function.\n")
    if not filename.endswith('.xlsx'):
        raise ValueError("Provided filename (%s) does not contain an Excel extenstion (.xlsx)." % (filename))
    if type(dfs) != list or type(output_sheet_names) != list:
        raise TypeError("Type of dfs: %s or type of output_sheet_names: %s is not list. Even if providing only a single element, ensure arguments are provided as lists." % (type(dfs), type(output_sheet_names)))
    if len(dfs) != len(output_sheet_names):
        raise ValueError("Length of dfs (%s) and length of output_sheet_names (%s) do not match." % (len(dfs),len(output_sheet_names)))
    
    i = -1
    for output in dfs:
        i += 1
        if "to_excel" not in dir(output):
            raise ValueError("Invalid Output at position %s (starting count at 0). dfs must be a pandas object with the property 'to_excel'." % (i))
        else:
            continue
    
    try:
        writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
        writer.save()
    except PermissionError:
        print("ERROR: Can't save the file while it is open. Please CLOSE the file and run again.")
    else:
        writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
        for i in range(len(dfs)):
            dfs[i].to_excel(writer,sheet_name=output_sheet_names[i])
            print("Writing DataFrame for Sheet: %s" % (output_sheet_names[i]))
        writer.save()
        print('Successfully wrote DataFrames to Excel file called: %s' % (filename))
    finally:
        writer.close()

def export_to_spss(filename, df, reset_index = False, DataType = 0, measure = 'scale', column_width = 8, align = 'right'):
    """
    Exports and saves the supplied pandas.DataFrame to a single SPSS data file (.sav).  This should be run at the end of a notebook.

    Required Arguments:
    filename (str)  -- file name of exported SPSS file. Must end with '.sav'
    df (pandas.DataFrame) -- pandas.DataFrame to be loaded into SPSS

    Optional Arguments:
    reset_index (bool) -- resets the index and adds it to the SPSS data file as a column. Useful in PPT is an index.  default = False (bool)
    DataType (int) -- Type of data (defined in Variable view of SPSS), default = 0 (Numerical)
    measure (str) -- Sets measure of data (defined in Variable view of SPSS), default = 'scale'
    column_width (int) -- Sets width of data column (defined in Variable view of SPSS), default = 8
    align (str) -- Sets cell alignment for data column (defined in Variable view of SPSS), default = 'right'
    """
    print("If you are suppling a '.groupby' pandas object as an 'dfs', it is recommended that you use '.unstack' method on the object for this function.\n")
    if not filename.endswith(".sav"):
        filename += ".sav"
        print("'filename' argument did not contain *.sav extension. 'filename' has been modified to: %s \n" % (filename))
    if reset_index:
        df = df.reset_index()
        print("Adding index as column to df.")
    else:
        print("reset_index set to False. Index column will not be included in output.")
    
    varNames = list(df.columns)
    varTypes = {}
    measureLevels = {}
    columnWidths = {}
    alignments = {}
    for var in varNames:
        varTypes.update({var:DataType})
        measureLevels.update({var:measure})
        columnWidths.update({var:column_width})
        alignments.update({var:align})

    try:
        with SavWriter(filename, varNames, varTypes, ioUtf8 = True, measureLevels = measureLevels) as writer:
            writer.writerows(df)
    except:
        raise ValueError("ERROR: Something went wrong. Check if the file is open.")
    else:
        print('\nSuccessfully wrote DataFrame to SPSS file called: %s' % (filename))

def import_from_eprime(raw_dir, formatted_dir, merged_output_name="", encoding = "UTF-16"):
    """
    doc string here
    """
    def __load_eprime(filename):
        def check(line):
            for drive in ['C','D','E','F','G','H']:
                drive += ":"
                if line.startswith(drive):
                    return True
            else:
                return False
            
        raw_file = raw_dir + os.sep + filename
        print("\nLoading %s..." % filename)
        
        #find where header starts
        with open(raw_file, 'r', encoding = encoding) as tsv:
            for line in tsv:
                if check(line):
                    if next(tsv).startswith("This file"):
                        print("Removing top two lines.")
                        skip = 2
                        break
                    else:
                        skip = 1
                        print("Removing top line.")
                        break
                else:
                    print("Not removing any lines.")
                    skip = 0
                    break
                
        #need to reopen or loop will start where we left off before
        #obtain header as an array for pandas
        with open(raw_file, 'r', encoding = encoding) as tsv:
            i = 0
            for line in tsv:
                if i == skip:
                    header = line.replace('[',"").replace(']',"").replace('.','_').replace('\n',"").split("\t")
                    print("Removed '[', ']' , '.', and '\\n' from Headers.")
                    break
                else:
                    i += 1

        #load in pandas
        df = pd.read_csv(raw_file, delimiter='\t',encoding=encoding,skiprows=skip + 1, names = header)

        #Clean up: Random Seed, Session Time UTC, resolving study list issue and session date issue
        if 'RandomSeed' in df.columns:
            df = df.drop(columns = "RandomSeed")
            print("Found and dropped RandomSeed column.")
                
        else:
            print("No RandomSeed columnn found. Check file to make sure there are no errors.")

        if 'SessionTimeUtc' in df.columns:
            df = df.drop(columns = "SessionTimeUtc")
            print("Found and dropped SessionTimeUtc column.")
        
        if 'SessionTime' in df.columns:
            df = df.drop(columns = "SessionTime")
            print("Found and dropped SessionTime column.")            
            
        df['SourceFile'] = filename
        for studylist in ['A','B','C','D','E','F']:
            studylist = "StudyList" + studylist
            if studylist in df.columns:
                df['StudyList'] = df[studylist]
                df= df.drop(columns = studylist)
                print('Found study list column: %s. Passed this column to "StudyList" and dropped.' % studylist)
                break
            else:
                continue
        else:
            print('Did not find a study list column.')

        if 'SessionDate' in df.columns:
            date = df['SessionDate'].unique()[0]
            if '-' in date:
                delimiter = '-'
            elif '/' in date:
                delimiter = '/'
                
            date = date.split(delimiter)
            df['Month'] = date[0]
            df['Day'] = date[1]
            df['Year'] = date[2]
            df = df.drop(columns = 'SessionDate')
            print("Found and dropped SessionDate column. Split into 'Month', 'Day', and 'Year' columns.")

        return df
    
    #Check if raw_dir exists
    if not os.path.exists(raw_dir):
        raise ValueError("Provided raw directory: '%s' does not exist." % raw_dir)

    #Check if formatted_dir exists
    if not os.path.exists(formatted_dir):
        raise ValueError("Provided formatted directory: '%s' does not exist." % formatted_dir)

    print("MAKE SURE E-RECOVERY FILES ARE NOT INCLUDED IN RAW PPT FILES DIRECTORY!!!!!!!!!\n")

    df = pd.DataFrame()
    for raw_file in os.listdir(raw_dir):
        filename = os.fsdecode(raw_file)
        if filename.endswith(".txt"):
            df = df.append(__load_eprime(filename), ignore_index = True)
            print("Successfully appended!")
    df.to_csv(formatted_dir + os.sep + merged_output_name + '.txt', sep='\t',index=False)
    return df
