def replace_in_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for i in range(994, min(len(lines), 1188)):
            lines[i] = lines[i].replace("S01", "S08")
        
        with open(outfilename, 'w') as file:
            file.writelines(lines)

        print(f"Successfully updated {outfilename}")
        
    except FileNotFoundError:
        print(f'Error: The file "{filename}" was not found.')
    except Exception as e:
        print(f"An error occurred: {e}")

filename = 'test.sm_70.cuasm'
outfilename = 'test.sm_70.cuasm' 
replace_in_file(filename)