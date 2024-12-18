import setuptools

setuptools.setup(
    name='retrieval_base', 
    version='1.1', 
    description='Base code for running retrievals.', 
    url='#',
    author='samderegt', 
    install_requires=[
        'numpy', 
        'pandas', 
        'matplotlib', 
        'corner', 
        'scipy', 
        'mpi4py', 
        'pymultinest', 
        #'species', 
        'PyAstronomy', 
        #'pyfastchem', 
        #'petitRADTRANS', 
        #'h5py', 
        ], 
    author_email='regt@strw.leidenuniv.nl', 
    packages=setuptools.find_packages(), 
    zip_safe=False, 
    )