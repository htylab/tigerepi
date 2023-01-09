from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: MIT License',
    "Operating System :: OS Independent"
]

setup(
     name='tigerepi',

     version='0.0.1',
     description='Processing EPI images based on deep-learning',
     long_description_content_type='text/markdown',
     url='https://github.com/htylab/tigerepi',

     author='Biomedical Imaging Lab, Taiwan Tech',
     author_email='',
     License='MIT',
     classifiers=classifiers,

     keywords='EPI brain tools',
     packages=find_packages(),
     entry_points={
        'console_scripts': [
            'tigerepi = tigerepi.bx:main',

        ]
    },
     python_requires='>=3.7',
     install_requires=[
             'numpy>=1.16.0',
             'nilearn>=0.9.1',
         ]
)
