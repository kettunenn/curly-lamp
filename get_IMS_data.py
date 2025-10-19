from tqdm import tqdm
import requests
import os
import patoolib


def download_data(url, output_dir):

    # Downloads a file from the given URL using streaming
    # Saves the file in the specified output directory
    # Returns the relative path where the file is stored

    file_name = os.path.basename(url)
    data_dir = os.path.join(output_dir, file_name)

    

    if not (os.path.isdir(data_dir.split('.',1)[0]) or os.path.isfile(data_dir)):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        os.makedirs(output_dir, exist_ok=True)
        
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            response.raise_for_status()
            
            with open(data_dir, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):                    
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    else:
        print("Data already exists")
    
    return data_dir



def data_extract(archive, output_dir):
    
    # Extracts a compressed archive (like .zip or .rar) using patoolib
    # Puts all extracted files into the given directory
    # Returns a list of all extracted file paths



    if os.path.isfile(archive):
        patoolib.extract_archive(archive, outdir=output_dir)

        extracted_files = []
        archive_dir = archive.split('.',1)[0]

        for (root,_,files) in os.walk(archive_dir):
            for name in files:
                    extracted_files.append(os.path.join(root, name))
        
        extracted_files.sort()
        
        return extracted_files

def clean():
    
    #TODO clean all archive files
    
    print()

def main():

    # Downloads a ZIP file containing data from NASA's IMS dataset
    # (URL: https://data.nasa.gov/docs/legacy/IMS.zip)
    #
    # After downloading, it extracts the main ZIP file,
    # then looks one directory down and extracts any archives found there.

    url = "https://data.nasa.gov/docs/legacy/IMS.zip"
    output_dir = "data/"
    
    data_dir = download_data(url,  output_dir)

    extracted_files = data_extract(data_dir,  output_dir)
    #os.remove(data_dir)
    
    data_extract(extracted_files[0], data_dir.split('.',1)[0])

    #TODO extract all data, remove the __MACOSX folder clean archives

    #TODO return a list of location of data

if __name__ == "__main__":
    main()
