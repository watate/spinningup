# Github Guide
## Introduction
This document aims to clear up any confusion I might have about how github works and how I should use github to maintain my code

## Current Workflow
1. Git clone to my laptop
	- Copy github link
	- git clone "insert github link"
2. Make changes to the local repository
	- Write code, do stuff
3. Commit changes using sublime merge
	- Stage changes you want to commit
	- Commit changes
4. Push changes to update github repository
	- git push -u origin, or
	- git push -u myrep

## Github Pages
### Testing my Jekyll site
bundle exec jekyll serve

### Error installing nokogiri for github pages
```
sudo gem install github-pages
```
This seems to solve the issue
