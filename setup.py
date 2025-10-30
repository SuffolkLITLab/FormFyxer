import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    install_requires=['pdfminer.six', 'pandas', 'pikepdf',
        'textstat', 'requests', 'numpy',
    'boxdetect', 'pdf2image', 'reportlab>=3.6.13', 'pdfminer.six',
    'opencv-python', 'ocrmypdf', 'eyecite', 'sigfig',
        'openai', 'python-dotenv', 'python-docx', 'tiktoken', 'transformers' 
    ],
    include_package_data = True
)
