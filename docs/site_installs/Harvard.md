# System info

    Rocky Linux release 8.7 (Green Obsidian)
    Linux holylogin02.rc.fas.harvard.edu 4.18.0-425.10.1.el8_7.x86_64 #1 SMP Thu Jan 12 16:32:13 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux

### Useful links

- [Harvard FAS RC User Guide](https://docs.rc.fas.harvard.edu/)
- [Running MPI jobs](https://docs.rc.fas.harvard.edu/kb/running-jobs/)

# Load modules

    module load python gcc/12.2.0-fasrc01 cmake/3.25.2-fasrc01 openmpi/4.1.5-fasrc01 gmp/6.2.1-fasrc01 mpfr/4.2.0-fasrc01

You may run `module -t list` to view loaded modules,
and `module purge` to unload all modules.

# Install SDPB

Use `RPATH` instead of `RUNPATH` in `mpicxx` linker, to fix shared library loading in SDPB:

    export OMPI_LDFLAGS="$(mpicxx --showme:link) -Wl,--disable-new-dtags"

## Boost

    wget https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.bz2
    tar -jxf boost_1_83_0.tar.bz2
    cd boost_1_83_0
    ./bootstrap.sh --prefix=$HOME/install --without-libraries=python
    ./b2 -j 16 --prefix=$HOME/install
    ./b2 --prefix=$HOME/install install
    cd ..

## Elemental

    git clone https://gitlab.com/bootstrapcollaboration/elemental.git
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/install -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc
    make && make install

## RapidJSON

    git clone https://github.com/Tencent/rapidjson.git
    cp -r rapidjson/include $HOME/install

## libarchive

    wget http://www.libarchive.org/downloads/libarchive-3.7.1.tar.xz
    tar -xf libarchive-3.7.1.tar.xz
    cd libarchive-3.7.1
    ./configure --prefix=$HOME/install
    make -j 16 && make install
    cd ..

## sdpb

    git clone https://github.com/davidsd/sdpb.git
    cd sdpb
    ./waf configure --boost-dir=$HOME/install --elemental-dir=$HOME/install --rapidjson-dir=$HOME/install --libarchive-dir=$HOME/install --prefix=$HOME/install/sdpb-master
    ./waf
    ./test/run_all_tests.sh
    ./waf install

# Install scalar_blocks

## Trilinos

    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DTrilinos_ENABLE_Sacado=ON -DTrilinos_ENABLE_Kokkos=OFF -DTrilinos_ENABLE_Teuchos=OFF -DCMAKE_INSTALL_PREFIX=$HOME/install ..

## scalar_blocks

    ./waf configure --prefix=/n/home09/wlandry/install --trilinos-dir=$HOME/install --boost-dir=$HOME/install

## Batch Scripts

    /n/home09/wlandry/sdpb/runs/TTTT_small.sh
    /n/home09/wlandry/scalar_blocks/runs/scalar_blocks.sh
