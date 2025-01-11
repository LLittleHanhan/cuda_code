def modify_compile():
    try:
        with open('compile.sh', 'r') as file:
            lines = file.readlines()
        
        ptxas_index = next((i for i, line in enumerate(lines) if "ptxas" in line), None)
        ptxas_index += 1      
        if ptxas_index is not None:
            lines = lines[ptxas_index:]
            lines = [line[3:] for line in lines]
            with open('compile.sh', 'w') as file:
                file.writelines(lines)

            print(f"Successfully modified compile.sh")
        else:
            print(f'Error: The file "compile.sh" does not contain the keyword "ptxas".')
    except FileNotFoundError:
        print(f'Error: The file "compile.sh" was not found.')
    except Exception as e:
        print(f"An error occurred: {e}")

modify_compile()