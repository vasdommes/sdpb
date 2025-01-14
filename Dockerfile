# syntax=docker/dockerfile:1

# This file contains three stages: 'build', 'install', 'test'
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#use-multi-stage-builds
#
# 'build' builds sdpb binaries. This image is heavy, because it contains all sources etc.
# 'install' contains only sdpb binaries (in /usr/local/bin/) + necessary dynamic libraries
# 'test' is based on 'install', but also copies sdpb/build/ and sdpb/test/ folders to /home/testuser/sdpb/
# - this allows to in order to run unit and integration tests by calling sdpb/test/run_all_tests.sh
# Final (lightweight) image is made from 'install' target.
#
# How to check that SDPB image works:
#
# docker build . -t sdpb
# docker run sdpb mpirun --allow-run-as-root sdpb --help
#
# How to run unit/integration tests to check that SDPB works correctly:
#
# docker build . -t sdpb-test --target test
# docker run sdpb-test ./test/run_all_tests.sh mpirun --oversubscribe
#
# Note: 'mpirun --oversubscribe' is necessary only if your environment has less than 6 CPUs available

# latest alpine release
FROM alpine:3.18 AS build

RUN apk add \
    cmake \
    g++ \
    git \
    make \
    python3 \
    unzip  \
    boost-dev \
    gmp-dev \
    libarchive-dev \
    libxml2-dev \
    mpfr-dev \
    openblas-dev \
    openmpi \
    openmpi-dev \
    rapidjson-dev
WORKDIR /usr/local/src
# Build Elemental
RUN git clone https://gitlab.com/bootstrapcollaboration/elemental.git && \
    cd elemental && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=/usr/local/ && \
    make && \
    make install
# Build SDPB from current sources
COPY . sdpb
RUN cd sdpb && \
    ./waf configure --elemental-dir=/usr/local --prefix=/usr/local && \
    python3 ./waf && \
    python3 ./waf install

# Take only sdpb binaries + load necessary dynamic libraries
FROM alpine:3.18 as install
RUN apk add \
    boost1.82-date_time \
    boost1.82-filesystem \
    boost1.82-iostreams \
    boost1.82-program_options \
    boost1.82-serialization \
    boost1.82-system \
    gmp \
    libarchive \
    libgmpxx \
    libstdc++ \
    libxml2 \
    mpfr \
    openblas \
    openmpi \
    openssh
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /usr/local/lib /usr/local/lib

# Separate test target, see
# https://docs.docker.com/language/java/run-tests/#multi-stage-dockerfile-for-testing
# Contains /home/testuser/sdpb/build and /home/testuser/sdpb/test folders,
# which is sufficient to run tests as shown below:
#
# docker build -t sdpb-test --target test
# docker run sdpb-test ./test/run_all_tests.sh mpirun --oversubscribe
FROM install as test
# Create testuser to run Docker non-root (and avoid 'mpirun --allow-run-as-root' warning)
RUN addgroup --gid 10000 testgroup && \
    adduser --disabled-password --uid 10000 --ingroup testgroup --shell /bin/sh testuser
WORKDIR /home/testuser/sdpb
COPY --from=build /usr/local/src/sdpb/build build
COPY --from=build /usr/local/src/sdpb/test test
RUN chown -R testuser test
USER testuser:testgroup

# Resulting image
FROM install
# TODO best practices suggest to run containter as non-root.
# https://github.com/dnaprawa/dockerfile-best-practices#run-as-a-non-root-user
# But this requires some extra work with permissions when mounting folders for docker run.