import os
import shutil

def load_dataset(images_path: str, added_new_images: bool = False) -> list:
    if added_new_images: 
        working_directory_path: str = os.path.dirname(images_path)
        extension: str = '.jpg'
        # Per comodità rinomino tutte le immagini su cui fare inferenza con nomi del tipo 1.jpg, 2.jpg, 3.jpg, ...
        files: list = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith('.')]
        num: int = 1
        for file in files:
            os.rename(file, str(num) + extension)
            num += 1
        # Poichè a seguito della ridenominazione le immagini vengono spostate al di fuori dalla cartella images, le riporto dentro.
        files: list = [os.path.join(working_directory_path, elem) for elem in os.listdir(working_directory_path) if not elem.startswith('.') and elem.endswith(extension)]
        for file in files:
            shutil.move(os.path.join(working_directory_path, file), images_path)
    # Creo il dataset con tutte le immagini su cui eseguire l'attacco FGSM.
    files: list = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith('.')]
    dataset = list()
    for file in files:
        dataset.append(file)
    # Opzionale: ordino le immagini del dataset in base al nome.
    dataset = sorted(dataset, key = lambda x: int(x[x.rfind('/') + 1 : x.rfind('.')]))
    return dataset