from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('requirements_extra.txt', 'r') as f:
    requirements_extra = f.read().splitlines()

setup(name='setpos',
      version='0.1.0',
      description='Setvalued part-of-speech-tagger build on top CoreNLP and TreeTagger',
      long_description=readme,
      author='Stefan Heid',
      author_email='stefan.heid@upb.de',
      python_requires='>=3.8',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
      ],
      url='https://github.com/stheid/SetPOS',
      install_requires=requirements,
      extras_require={'extra': requirements_extra},
      license="GNU General Public License v3"
      )
