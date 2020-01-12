#### Directory structure for generic projects
Empty structure repo.  Clone and rename.
Use subdirectories as needed. 
Default gitignore behaviour will ignore everyhting under "local" directory and all files with .tmp extension.  This would be a good place to store large files or temporary experimental results.

#### Setup the repo
```
REPO_NAME=$(basename $(pwd))
echo $REPO_NAME

conda create -y --name $REPO_NAME python==3.7
conda activate $REPO_NAME
conda install sphinx
```

#### to into your project repo
```
(skeleton) jkhouja@Judes-MacBook-Pro: ~/workspace $cd test
```

#### Create a docs directory
```
(skeleton) jkhouja@Judes-MacBook-Pro: ~/workspace/repo/test $mkdir docs
(skeleton) jkhouja@Judes-MacBook-Pro: ~/workspace/repo/test $cd docs
```

#### Run sphinx-quickstart from docs
```
(skeleton) jkhouja@Judes-MacBook-Pro: ~/workspace/repo/test/docs $sphinx-quickstart
Welcome to the Sphinx 2.3.1 quickstart utility.

Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets).

Selected root path: .

You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]: y

The project name will occur in several places in the built documentation.
> Project name: Test
> Author name(s): Jude
> Project release []: Jan 2020

If the documents are to be written in a language other than English,
you can select a language here by its language code. Sphinx will then
translate text that it generates into that language.

For a list of supported codes, see
https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
> Project language [en]:

Creating file ./source/conf.py.
Creating file ./source/index.rst.
Creating file ./Makefile.
Creating file ./make.bat.

Finished: An initial directory structure has been created.

You should now populate your master file ./source/index.rst and create other documentation
source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.
```
#### Update conf.py

1. Add proper imports to add your modules
```
sys.path.insert(0, os.path.abspath('../..'))
import src
```
1. Add 'sphinxcontrib.napoleon' to extensions:
```
extensions = ['sphinxcontrib.napoleon']
```
### Run apidoc to generate rst files
source/ is the output directory of rst files
../src/ is the directory of all your sourcecode
```
sphinx-apidoc -f -o source/ ../src/
Creating file source/text.rst.
Creating file source/utils.rst.
Creating file source/modules.rst.
```
### Build documentations
```
(skeleton) jkhouja@Judes-MacBook-Pro: ~/workspace/repo/test/docs $sphinx-build -b html source/ build/
Running Sphinx v2.3.1
building [mo]: targets for 0 po files that are out of date
building [html]: targets for 4 source files that are out of date
updating environment: [new config] 4 added, 0 changed, 0 removed
reading sources... [100%] utils
WARNING: autodoc: failed to import module 'text'; the following exception was raised:
No module named 'text'
WARNING: autodoc: failed to import module 'utils'; the following exception was raised:
No module named 'utils'
looking for now-outdated files... none found
pickling environment... done
checking consistency... /Users/jkhouja/workspace/repo/test/docs/source/modules.rst: WARNING: document isn't included in any toctree
done
preparing documents... done
writing output... [100%] utils
generating indices...  genindexdone
writing additional pages...  searchdone
copying static files... ... done
copying extra files... done
dumping search index in English (code: en)... done
dumping object inventory... done
build succeeded, 3 warnings.

The HTML pages are in build.
```

