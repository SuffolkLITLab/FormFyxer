import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# We can't simply include this as an install_requires, because pypi won't allow
# projects with github dependencies to be hosted there.
#class InstallSpacyModelCommand(install):
#    def run(self):
#        install.run(self)
#        import spacy
#        print("Downloading word2vec model en_core_web_sm")
#        spacy.cli.download('en_core_web_sm')


setuptools.setup(
    name='formfyxer',
    version='0.3.0a3',
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
    install_requires=['spacy', 'pdfminer.six', 'pandas', 'pikepdf',
        'textstat', 'requests', 'numpy', 'scikit-learn', 'networkx', 'joblib',
        'nltk', 'boxdetect', 'pdf2image', 'reportlab>=3.6.13', 'pdfminer.six',
        'opencv-python', 'ocrmypdf', 'eyecite', 'passivepy>=0.2.16', 'sigfig',
        'typer>=0.4.1,<0.5.0', # typer pre 0.4.1 was broken by click 8.1.0: https://github.com/explosion/spaCy/issues/10564
        'openai', 'python-docx', 'tiktoken', 'transformers' 
    ],
    #cmdclass={
    #  'install': InstallSpacyModelCommand,
    #},
    include_package_data = True
)
