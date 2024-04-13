## Data

For now, the data can be found in https://drive.google.com/drive/folders/1OG0NaPqlbzNlvG83L0LQMMVsw8jftEsz
Download the zipfile and create a `data` folder with its elements. 
Then add this folder and the `plots` folder to the project folder.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

All packages are listed in [the requirements file](requirements.txt). To install these packages, you can use the following command if you are using `pip`:

```bash
pip install -r requirements.txt
```

## Project Structure

This section provides a detailed overview of the project's directory and file structure, explaining the purpose and contents of each part.

### `webscraping/`

- Scripts designed to extract data from the web, facilitating the data collection process for analysis. They can be viewed as an inspiration for future webscrapping, as they will be very hard to use again. 

### `src/`

- The source code for the core functionality of the project.
  - [`src/Processing/`](src/Processing/): All the functions linked use to filter texts on the theme of BigTechs, and text cleaning functions. The ```clean()``` 

### `processing/`

- Scripts and modules responsible for data processing and manipulation.
  - `demo_and_requirements/`: Includes scripts showcasing processing capabilities and additional requirements if applicable.

### `notebooks/`

- Contains Jupyter notebooks that demonstrate or test various features of the project.
  - `main.ipynb`: A notebook that provides a comprehensive demo of the project's capabilities.

### `plots/`

- This directory houses all graphical outputs generated by the project. It includes:
  - `Polarization/`: Subdirectory containing plots specific to polarization studies or results.



