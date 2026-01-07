#!/usr/bin/env python3
import re
import sys
import argparse
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
PYPROJECT_FILE = ROOT_DIR / "pyproject.toml"
README_FILE = ROOT_DIR / "README.md"

def get_current_version():
    content = PYPROJECT_FILE.read_text()
    match = re.search(r'version = "(\d+\.\d+\.\d+)"', content)
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)
    return match.group(1)

def bump_version(current_version, part):
    major, minor, patch = map(int, current_version.split('.'))
    if part == 'major':
        major += 1
        minor = 0
        patch = 0
    elif part == 'minor':
        minor += 1
        patch = 0
    elif part == 'patch':
        patch += 1
    return f"{major}.{minor}.{patch}"

def update_files(new_version):
    # 1. Update pyproject.toml
    content = PYPROJECT_FILE.read_text()
    new_content = re.sub(r'version = "\d+\.\d+\.\d+"', f'version = "{new_version}"', content)
    PYPROJECT_FILE.write_text(new_content)
    print(f"Updated pyproject.toml to {new_version}")

    # 2. Update README.md (Installation URL)
    readme_content = README_FILE.read_text()
    # Regex to find git+https://...@vX.Y.Z
    # We look for the pattern and replace the tag part
    pattern = r'(git\+https://github\.com/[^/]+/[^.]+\.git)@v\d+\.\d+\.\d+'
    
    if re.search(pattern, readme_content):
        new_readme = re.sub(pattern, f'\\1@v{new_version}', readme_content)
        README_FILE.write_text(new_readme)
        print(f"Updated README.md installation link to v{new_version}")
    else:
        print("Warning: Could not find git installation link with version tag in README.md")

def git_operations(new_version):
    tag_name = f"v{new_version}"
    try:
        # Stage files
        subprocess.run(["git", "add", "pyproject.toml", "README.md"], check=True)
        
        # Commit
        subprocess.run(["git", "commit", "-m", f"Release {tag_name}"], check=True)
        
        # Tag
        subprocess.run(["git", "tag", tag_name], check=True)
        
        print(f"\nSUCCESS: Created git tag {tag_name}")
        print(f"Now push to origin:\n  git push origin main\n  git push origin {tag_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automate release versioning")
    parser.add_argument("part", choices=['major', 'minor', 'patch'], help="Part of version to bump")
    args = parser.parse_args()

    current_ver = get_current_version()
    new_ver = bump_version(current_ver, args.part)
    
    print(f"Bumping version: {current_ver} -> {new_ver}")
    
    confirm = input("Continue? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    update_files(new_ver)
    git_operations(new_ver)

if __name__ == "__main__":
    main()
