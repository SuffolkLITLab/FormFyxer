# FormFyxer
A Python package with a collection of function for learning about and pre-processing pdf forms, with an eye towards interoperability with the Legal Innovation and Technology Lab's [Document Assembly Line Project](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/).

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FormFyxer.
Rerun this command to check for and install updates.
```bash
pip install git+https://github.com/SuffolkLITLab/FormFyxer
```

## Usage
Features:
* FormFyxer.parse_form  --> Read in a pdf, pull out basic stats, attempt to normalize its form fields, and re-write the file with the new fields (if rewrite=1).

#### Demo of some of the features:
```python
import formfyxer as ff

ff.parse_form("sample.pdf",title="Sample Form",jur="CA",cat="Housing",normalize=1,use_spot=0,rewrite=0)
```

## License
[MIT](https://github.com/SuffolkLITLab/FormFyxer/blob/main/LICENSE)
