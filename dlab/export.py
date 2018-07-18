import pandas as pd
from savReaderWriter import * 

def export_to_excel(filename, dfs, output_sheet_names):
    """
    This function should be run LAST on the Notebook.
    filename -> file name of exported excel file
    dfs -> pandas DataFrame(s) to be exported to Excel provided as a [list]. To write mulitiple sheets in one Excel File, provide each DataFrame in the [list]. 
    output_sheet_names -> Name of the sheet in the exported Excel file. For each DataFrame provided in 'dfs', a sheet name must also be supplied as a [list].
    """
    print("If you are suppling a '.groupby' pandas object as an 'dfs', it is recommended that you use '.unstack' method on the object for this function.\n")
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
        writer = pd.ExcelWriter(filename)
        writer.save()
    except PermissionError:
        print("ERROR: Can't save the file while it is open. Please CLOSE the file and run again.")
    else:
        writer = pd.ExcelWriter(filename)
        for i in range(len(dfs)):
            dfs[i].to_excel(writer,sheet_name=output_sheet_names[i])
            print("Writing DataFrame for Sheet: %s" % (output_sheet_names[i]))
        writer.save()
        print('Successfully wrote DataFrames to Excel file called: %s' % (filename))
    finally:
        writer.close()

def export_to_spss(filename, df, reset_index = False, DataType = 0, measure = 'scale', column_width = 8, align = 'right'):
    """
    This function should be run LAST on the Notebook.
    filename  -> file name of exported SPSS file
    df -> pandas.DataFrame to be loaded into SPSS
    DataType -> Type of data (defined in Variable view of SPSS), default = 0 (Numerical)
    measure -> Sets measure of data (defined in Variable view of SPSS), default = 'scale'
    column_width -> Sets width of data column (defined in Variable view of SPSS), default = 8
    align -> Sets cell alignment for data column (defined in Variable view of SPSS), default = 'right'
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
