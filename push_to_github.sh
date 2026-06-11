#!/bin/bash
# Clean up and optimize markdown
mv "Rag ai docs project.md" README.md
echo "# Rag-ai-docs-project" > temp.md
echo "" >> temp.md
cat README.md >> temp.md
mv temp.md README.md

# Initialize and push to GitHub
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/hardik79999/Rag-ai-docs-project.git
git push -u origin main
