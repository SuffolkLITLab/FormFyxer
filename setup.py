import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class InstallSpacyModelCommand(install):
    def run(self):
        install.run(self)
        import spacy
        print("Downloading word2vec model en_core_web_lg")
        spacy.cli.download('en_core_web_lg')


setuptools.setup(
    name='formfyxer',
    version='0.0.8a1',
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
    cmdclass={
      'install': InstallSpacyModelCommand,
    },
    include_package_data = True
)

