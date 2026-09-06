from setuptools import find_packages, setup

setup(
    name="deepseekmoe_dynamic_routing_algorithms",
    version="2.1.0",
    packages=find_packages(),
    url="https://github.com/MisterFOURXXX/deepseekmoe_dynamic_routing_algorithms",
    license="Apache 2.0",
    author="Apiwit Karnjanavivin",
    author_email="Apiwit.Karnjanavivin@gisma-student.com",
    description="Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms",
    keywords=["deepseek", "moe", "routing"],
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True,
)