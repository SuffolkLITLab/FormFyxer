import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='formfyxer',
    version='0.0.4',
    author='Suffolk LIT Lab',
    author_email='litlab@suffolk.edu',
    description='A tool for learning about and pre-processing pdf forms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SuffolkLITLab/FormFyxer',
    project_urls = {
        "Bug Tracker": "https://github.com/SuffolkLITLab/FormFyxer/issues"
    },
    license='MIT',
    packages=['formfyxer'],
    install_requires=['spacy',  'PyPDF2',  'pikepdf',  'textstat',  'requests',  'numpy',  'sklearn', 'joblib',  'nltk', 'boxdetect', 'pdf2image', 'reportlab', 'pdfminer.six', 'opencv-python'],
    include_package_data = True
)
