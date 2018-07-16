import pandas as pd

def export_to_excel(filename, outputs, output_sheet_names):
    print("If you are suppling a '.groupby' pandas object as an 'outputs', it is recommended that you use '.unstack' method on the object for this function.\n")
    if not filename.endswith('.xlsx'):
        raise ValueError("Provided filename (%s) does not contain an Excel extenstion (.xlsx)." % (filename))
    if type(outputs) != list or type(output_sheet_names) != list:
        raise TypeError("Type of outputs: %s or type of output_sheet_names: %s is not list. Even if providing only a single element, ensure arguments are provided as lists." % (type(outputs), type(output_sheet_names)))
    if len(outputs) != len(output_sheet_names):
        raise ValueError("Length of outputs (%s) and length of output_sheet_names (%s) do not match." % (len(outputs),len(output_sheet_names)))
    
    i = -1
    for output in outputs:
        i += 1
        if "to_excel" not in dir(output):
            raise ValueError("Invalid Output at position %s (starting count at 0). Outputs must be a pandas object with the property 'to_excel'." % (i))
        else:
            continue
    
    try:
        writer = pd.ExcelWriter(filename)
        writer.save()
    except PermissionError:
        print("ERROR: Can't save the file while it is open. Please CLOSE the file and run again.")
    else:
        writer = pd.ExcelWriter(filename)
        for i in range(len(outputs)):
            outputs[i].to_excel(writer,sheet_name=output_sheet_names[i])    
        writer.save()
        print('Successfully wrote DataFrames to Excel file called: %s' % (filename))
    finally:
        writer.close()