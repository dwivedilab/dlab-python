import os
import pandas as pd

def import_data(mode = "merge", source = "eprime", raw_dir = "", formatted_dir = "", merged_output_name = "merged", encoding = "UTF-16"):
    #Define local functions
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
    
    def __add_eprime():
        formatted_files = []
        for formatted_file in os.listdir(formatted_dir):
            filename = os.fsdecode(formatted_file)
            if filename.endswith(".txt"):
                formatted_files.append(filename)

        print(formatted_files)
        for raw_file in os.listdir(raw_dir):
            filename = os.fsdecode(raw_file)
            if filename.endswith(".txt") and filename not in formatted_files:
                __load_eprime(filename).to_csv(formatted_dir + os.sep + filename, sep='\t',index=False)
                df = df.append(__load_eprime(filename), ignore_index = True)                
                print("Successfully exported!")
        return df

    def __merge_eprime():
        df = pd.DataFrame()
        for raw_file in os.listdir(raw_dir):
            filename = os.fsdecode(raw_file)
            if filename.endswith(".txt"):
                df = df.append(__load_eprime(filename), ignore_index = True)
                print("Successfully appended!")
        df.to_csv(formatted_dir + os.sep + merged_output_name + '.txt', sep='\t',index=False)
        return df

    def __add_psychopy():
        pass
    
    def __merge_psychopy():
        pass
        
    #Check if valid source type and create load_type dict
    if source == "eprime":            
        print("MAKE SURE E-RECOVERY FILES ARE NOT INCLUDED IN RAW PPT FILES DIRECTORY!!!!!!!!!\n")
        load_type = {"add":__add_eprime, "merge":__merge_eprime}
    elif source == "psychopy":
        load_type = {"add":__add_psychopy, "merge":__merge_psychopy}
        raise ValueError('PsychoPy methods not yet implemented')
    else:
        raise ValueError("Provided source: '%s' is invalid. \nImport is only implemented for 'eprime' and 'psychopy'. \nProvide one of these as the argument for source.")

    #Check if raw_dir exists
    if not os.path.exists(raw_dir):
        raw_dir = "Raw PPT Files"
        if os.path.exists("Raw PPT Files"):
            print("Setting raw directory to: Raw PPT Files\n")
        else:
            raise ValueError("Provided raw directory: '%s', and default raw directory 'Raw PPT Files' do not exist." % raw_dir)
        
    #Check if formatted_dir exists and if not create
    if not os.path.exists(formatted_dir):
        formatted_dir = "Formatted PPT Files"
        if os.path.exists("Formatted PPT Files"):
            print("Setting formatted directory to: Formatted PPT Files")
        else:
            print("Provided formatted directory: '%s', and default formatted directory 'Formatted PPT Files' do not exist." % formatted_dir)
            print("Creating 'Formatted PPT Files' directory.")
            os.makedirs(formatted_dir)
            print("Successfully created directory!\n")
        
    #Check if valid mode and execute function
    if mode in load_type:
        return load_type[mode]()
    else:
        raise ValueError("Provided mode: '%s' is invalid. \nImplemented import modes are 'add' and 'merge'. \nProvide one of these as the argument for source.")
