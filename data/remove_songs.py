import os
import glob

log_file = 'removed_songs.txt'

def remove_songs(folder_path, song_title, log_file=log_file, ask=True):
    # Search for files with the given title in the specified folder
    files = []
    for root, _, filenames in os.walk(folder_path):
        files.extend(glob.glob(os.path.join(root, song_title + '.*')))

    if len(files) == 0:
        print("No matching files found.")
        return

    print("Matching files found:")
    for file_path in files:
        print(file_path)

    if ask:
        answer = input("Do you want to delete these files? (yes/no): ")
    else:
        answer = "yes"

    if answer.lower() == "yes":
        removed_songs = []

        for file_path in files:
            try:
                os.remove(file_path)
                removed_songs.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error while deleting file: {file_path}")
                print(f"Error message: {str(e)}")

        if len(removed_songs) > 0:
            print("The following songs have been removed:")
            
            for song in removed_songs:
                print(song)
            for file_path in files:
                # Append removed song title to the log file
                with open(log_file, 'a') as log:
                    log.write(file_path + '\n')
        else:
            print("No songs have been removed.")
    else:
        print("No files have been deleted.")


if __name__ == '__main__':
    folder_path = './'
    song_title = input("Enter the title of the song: ")

    remove_songs(folder_path, song_title, log_file)
