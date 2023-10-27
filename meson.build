project('LMA','cpp',default_options : ['cpp_std=c++17'])

compiler = meson.get_compiler('cpp')
compiler_id = compiler.get_id()

subproject_default_options = ['enable_unit_tests=false']

includes = include_directories('.')

openmp = dependency('openmp') # meson builtin
fftw = dependency('fftw3')
eigen = dependency('eigen3')
netcdf = dependency('netcdf', language: 'cpp', version : '>=4.3.1')

executable('main', 'Source/main.cpp', dependencies: [openmp, fftw, eigen, netcdf], link_args: ['-lnetcdf_c++4'] )
executable('approximation_test', 'Source/approximation_test.cpp', dependencies: [openmp, fftw, eigen, netcdf], link_args: ['-lnetcdf_c++4'] )
