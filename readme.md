# paJAMeter Server

A Parameter Server implementation using the JAMScript language

The jamscript language can be installed [here](https://github.com/citelab/JAMScript): to install, run install.sh; JAMScript has worked on MacOS and Linux (Arch, Ubuntu, Raspian). Note that on some macos devices, brew can install mosquitto to a directory outside of the system path, so it may need to be added to PATH, C_INCLUDE_PATH, and/or LIBRARY_PATH manually; the jamtools should also be added to the PATH.

Once installed, the parameter server can be compiled with `jamc -d pajameter.*`, which will produce a file `pajameter.jxe` if JAMScript was installed correctly.

Before running, the global mosquitto broker should be started using the configuration files found in the JAMScript source; run `mosquitto -c temp/mosq18830.conf`. This makes the connections faster and more reliable, but is not strictly necessary.

To run, use the `jamrun` command.

`jamrun pajameter.jxe --app=ps --fog` will start the parameter server; this should be done once.

`jamrun pajameter.jxe --app=ps` will start a worker; this can be done as many times as desired.

The sequential reference implementation can be compiled through the makefile.
