import os

def print_files_recursively(directory_path):
    """
    Recursively prints the filename and content of all files in a given directory.

    Args:
        directory_path (str): The path to the starting directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory path.")
        return

    print(f"--- Traversing directory: {directory_path} ---")

    # os.walk yields a 3-tuple (dirpath, dirnames, filenames) for each directory.
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(f"\n*** File: {file_path} ***")
            try:
                # Open and read the file's content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content)
            except IOError as e:
                print(f"Could not read file {file_path}: {e}")
            except UnicodeDecodeError:
                print(f"Could not read file {file_path} (not a text file or unsupported encoding).")

# Example usage (replace '.' with the path to your desired directory)
# '.' refers to the current directory where the script is run.
print_files_recursively('.')
