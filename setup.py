from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="I3DM",
    version="0.1.0",
    author="Your Name",
    description="Your project description",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires=">=3.8",
)