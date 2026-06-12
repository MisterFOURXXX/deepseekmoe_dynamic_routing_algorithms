from setuptools import setup, find_packages

setup(
    name="DeepSeekMoE-Dynamic-Routing-Algorithms",
    version="1.0.0",
    packages=find_packages(),
    url="https://github.com/MisterFOURXXX/deepseekmoe_dynamic_routing_algorithms",
    license="Apache 2.0",
    author="Mohammad Mahdavi, Apiwit Karnjanavivin",
    author_email=["mohammad.mahdavi@gisma.com", "Apiwit.Karnjanavivin@gisma-student.com"],  # Added comma here
    description="Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms",
    keywords=["", "", "", ""],
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True,
)