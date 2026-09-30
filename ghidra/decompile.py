#!/usr/bin/env python2
# -*- coding:utf-8 -*-

"""
Python Script used to communicate with Ghidra's API.
It will decompile all the functions of a defined binary and
save results into decompiled_output.c

The code is pretty straightforward, it includes comments and it is easy to understand.
This will help people that is starting with Automated Malware Analysis
using Headless scripts with Ghidra.

Modified from https://github.com/galoget/ghidra-headless-scripts
"""

from __future__ import print_function
import sys
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
import __main__ as ghidra_app
args = ghidra_app.getScriptArgs()
if len(args) != 1:
    raise ValueError("Expected one output path")

# Communicates with Decompiler Interface
decompinterface = DecompInterface()

# Open Current Program
decompinterface.openProgram(currentProgram);

# Get Binary Functions
functions = currentProgram.getFunctionManager().getFunctions(True)

# Prints Current Python version (2.7)
print("Current Python version: " + str(sys.version))

# Iterates through all functions in the binary and decompiles them
# Then prints the Pseudo C Code

try:
    with open(args[0], "w") as output_file:
        for function in functions:
            result = decompinterface.decompileFunction(function, 60, ConsoleTaskMonitor())
            if not result.decompileCompleted() or result.getDecompiledFunction() is None:
                print("Failed to decompile " + str(function) + ": " + str(result.getErrorMessage()))
                continue
            output_file.write("// Function: " + str(function.getName()) + "\n")
            output_file.write(result.getDecompiledFunction().getC() + "\n\n")
finally:
    decompinterface.dispose()
