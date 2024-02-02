"""wip
"""
from setuptools import find_packages, setup

IGNORE_ITEMS = ["-e .", "-i https://pypi.org/simple", ""]

__VERSION__ = "0.0.0"
REPO_NAME = "Wine-Quality-Predictor"
SRC_REPO = "src"
AUTHOR_NAME = "Udit Kumar Chatterjee"
AUTHOR_EMAIL = "quantumudit@gmail.com"
AUTHOR_GITHUB_USERNAME = "quantumudit"
SHORT_DESCRIPTION = "A small python package for ML app"

with open("README.md", "r", encoding="utf-8") as readme:
    LONG_DESCRIPTION = readme.read()


def get_requirements(file_path: str) -> list[str]:
    """_summary_

    Args:
        file_path (str): _description_

    Returns:
        list[str]: _description_
    """
    with open(file_path, "r", encoding="utf-8") as f:
        contents = [item.strip() for item in f.readlines()]
        requirements = [item for item in contents if item not in IGNORE_ITEMS]
        return requirements


setup(
    name=SRC_REPO,
    version=__VERSION__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_GITHUB_USERNAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_GITHUB_USERNAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("./requirements.txt")
)
